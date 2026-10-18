#!/usr/bin/env python3
"""
evaluate_llm.py

Option A: Single-file LLM evaluation pipeline for:
 - Relevance & Completeness
 - Hallucination / Factual Accuracy (context support)
 - Latency & Cost estimation

Usage:
    python evaluate_llm.py --conv sample-chat-conversation-01.json \
                           --context sample_context_vector_01.json \
                           --out report.json

This script is intentionally dependency-friendly:
 - If sentence-transformers is installed, uses it for semantic similarity.
 - Otherwise falls back to sklearn TF-IDF cosine similarity.

Author: Siddharth's helper (prepare to push to GitHub)
PEP8-compatible-ish.
"""
import argparse
import json
import time
import math
import os
from typing import List, Dict, Any, Tuple

# Try semantic embedder first
EMBEDDER_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer, util

    EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    EMBEDDER_AVAILABLE = True
except Exception:
    # Fallback to sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    EMBEDDER = None

# -------------------------
# Helper utilities
# -------------------------
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def approx_tokens_from_text(text: str) -> int:
    # heuristic: 1 token ≈ 4 characters (approx)
    return max(1, math.ceil(len(text) / 4.0))


# -------------------------
# Similarity functions
# -------------------------
def semantic_similarity(a: List[str], b: List[str]) -> List[List[float]]:
    """
    Return cosine similarity matrix between lists a and b.
    Uses sentence-transformers if available, else TF-IDF.
    """
    if EMBEDDER_AVAILABLE:
        emb_a = EMBEDDER.encode(a, convert_to_tensor=True)
        emb_b = EMBEDDER.encode(b, convert_to_tensor=True)
        sim_matrix = util.cos_sim(emb_a, emb_b).cpu().tolist()
        return sim_matrix
    else:
        # TF-IDF fallback
        vectorizer = TfidfVectorizer(stop_words="english").fit(a + b)
        vecs_a = vectorizer.transform(a)
        vecs_b = vectorizer.transform(b)
        sim = cosine_similarity(vecs_a, vecs_b)
        return sim.tolist()


# -------------------------
# Core evaluation logic
# -------------------------
def extract_last_user_and_response(conversation_json: Dict[str, Any]) -> Tuple[str, str, Dict]:
    """
    Expecting conversation JSON that contains a list of messages.
    Each message: { role: "user"|"assistant"|"system", content: "..." }
    We take the last user message and the assistant response that follows.
    If conversation format different, try common fallbacks.
    """
    messages = conversation_json.get("messages") or conversation_json.get("chat") or conversation_json
    if isinstance(messages, dict):
        # maybe single-turn dict with keys 'user' and 'assistant'
        user = messages.get("user", "")
        assistant = messages.get("assistant", "")
        meta = {}
        return user, assistant, meta

    last_user = ""
    last_assistant = ""
    for msg in messages[::-1]:
        role = msg.get("role", "").lower()
        if role == "assistant" and not last_assistant:
            last_assistant = msg.get("content", "")
        elif role == "user" and not last_user:
            last_user = msg.get("content", "")
        if last_user and last_assistant:
            break
    meta = {"total_messages": len(messages)} if isinstance(messages, list) else {}
    return last_user, last_assistant, meta


def score_relevance_and_completeness(user_text: str, assistant_text: str, context_texts: List[str]) -> Dict[str, Any]:
    """
    Relevance: semantic similarity between assistant_text and user_text + context
    Completeness: coverage of user's explicit asks found in assistant_text (keyword heuristic)
    """
    t0 = time.perf_counter()
    # Similarity between assistant and user
    sim_matrix = semantic_similarity([assistant_text], [user_text])
    sim_assistant_user = float(sim_matrix[0][0])

    # Similarity between assistant and each context chunk: take max
    if context_texts:
        sim_matrix_ctx = semantic_similarity([assistant_text], context_texts)
        # sim_matrix_ctx is 1 x N
        max_ctx_sim = max(sim_matrix_ctx[0])
    else:
        max_ctx_sim = 0.0

    # Completeness heuristic: check presence of keywords from user in assistant
    # Extract nouns/keywords naive: split by whitespace, lower, remove short words.
    user_words = [w.lower().strip(".,?;:()[]") for w in user_text.split() if len(w) > 3]
    # keep unique top 10
    keywords = list(dict.fromkeys(user_words))[:10]
    found = sum(1 for k in keywords if k in assistant_text.lower())
    completeness = (found / max(1, len(keywords))) if keywords else 0.0

    elapsed = time.perf_counter() - t0
    return {
        "similarity_assistant_user": sim_assistant_user,
        "max_similarity_with_context": max_ctx_sim,
        "completeness": completeness,
        "keywords_checked": keywords,
        "keywords_found": found,
        "time_seconds": elapsed,
    }


def detect_hallucination(assistant_text: str, context_texts: List[str], threshold: float = 0.45) -> Dict[str, Any]:
    """
    Simple heuristic: split assistant into sentences; for each declarative sentence,
    compute max similarity against context. If below threshold -> flag as unsupported.
    Returns list of unsupported sentences and a hallucination_score (fraction unsupported).
    """
    t0 = time.perf_counter()
    import re

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', assistant_text) if s.strip()]
    if not sentences:
        return {"unsupported_sentences": [], "hallucination_score": 0.0, "time_seconds": 0.0}

    sim_matrix = semantic_similarity(sentences, context_texts if context_texts else [""]) if context_texts else [[0.0] * len(sentences)]

    unsupported = []
    for i, sent in enumerate(sentences):
        max_sim = max(sim_matrix[i]) if context_texts else 0.0
        if max_sim < threshold:
            unsupported.append({"sentence": sent, "max_context_similarity": float(max_sim)})

    hallucination_score = len(unsupported) / len(sentences)
    elapsed = time.perf_counter() - t0
    return {"unsupported_sentences": unsupported, "hallucination_score": hallucination_score, "time_seconds": elapsed}


def measure_latency_and_cost(assistant_text: str,
                             user_text: str,
                             instrument: Dict[str, Any]) -> Dict[str, Any]:
    """
    instrument: configuration dict:
      - token_price_per_1k: float (e.g., 0.002 for 0.002 USD per 1k tokens)
      - include_context_tokens: bool
      - context_texts: list[str]
    """
    t0 = time.perf_counter()
    assistant_tokens = approx_tokens_from_text(assistant_text)
    user_tokens = approx_tokens_from_text(user_text)
    context_tokens = 0
    if instrument.get("include_context_tokens") and instrument.get("context_texts"):
        context_tokens = sum(approx_tokens_from_text(c) for c in instrument.get("context_texts"))

    total_tokens = assistant_tokens + user_tokens + context_tokens
    price_per_1k = instrument.get("token_price_per_1k", 0.0)
    estimated_cost = (total_tokens / 1000.0) * price_per_1k

    # latency measurement: simulated as the time to run similarity/hallucination steps around this function call.
    elapsed = time.perf_counter() - t0
    return {
        "assistant_tokens": assistant_tokens,
        "user_tokens": user_tokens,
        "context_tokens": context_tokens,
        "total_tokens_estimate": total_tokens,
        "estimated_cost": estimated_cost,
        "time_seconds": elapsed,
    }
#may it contain some technical error so if you find them please let me know

# -------------------------
# Main evaluate function
# -------------------------
def evaluate(conversation_json: Dict[str, Any], context_json: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    user_text, assistant_text, meta = extract_last_user_and_response(conversation_json)

    # context_json is expected to be list of dicts with 'id', 'content', 'score' etc.
    context_texts = []
    if isinstance(context_json, dict):
        # Try common structures
        if "chunks" in context_json:
            context_texts = [c.get("content", "") for c in context_json["chunks"]]
        elif "results" in context_json:
            context_texts = [r.get("text", "") or r.get("content", "") for r in context_json["results"]]
        else:
            # try values
            context_texts = [v for v in (context_json.get("contexts") or [])]
    elif isinstance(context_json, list):
        # assume list of chunks
        context_texts = [c.get("content", c.get("text", "")) if isinstance(c, dict) else str(c) for c in context_json]
    else:
        context_texts = []

    report = {
        "meta": meta,
        "timestamp": time.time(),
        "input_user_text": user_text,
        "input_assistant_text": assistant_text,
    }

    # Track total evaluation time
    t_start = time.perf_counter()

    # Relevance & Completeness
    rel = score_relevance_and_completeness(user_text, assistant_text, context_texts)
    report["relevance_completeness"] = rel

    # Hallucination detection
    hall = detect_hallucination(assistant_text, context_texts, threshold=config.get("hallucination_threshold", 0.45))
    report["hallucination"] = hall

    # Latency & cost
    instrument = {
        "token_price_per_1k": config.get("token_price_per_1k", 0.0),
        "include_context_tokens": config.get("include_context_tokens", True),
        "context_texts": context_texts,
    }
    cost = measure_latency_and_cost(assistant_text, user_text, instrument)
    report["latency_cost"] = cost

    t_total = time.perf_counter() - t_start
    report["evaluation_time_seconds"] = t_total

    # Aggregate score (simple weighted average)
    # similarity_assistant_user [0..1], completeness [0..1], (1 - hallucination_score) [0..1]
    w_sim = 0.45
    w_comp = 0.35
    w_hall = 0.20
    agg = (w_sim * rel["similarity_assistant_user"]
           + w_comp * rel["completeness"]
           + w_hall * (1.0 - hall["hallucination_score"]))
    report["aggregate_score"] = agg

    # Add human-readable verdicts
    report["verdicts"] = {
        "relevance": "good" if rel["similarity_assistant_user"] > 0.6 else ("weak" if rel["similarity_assistant_user"] > 0.35 else "poor"),
        "completeness": "complete" if rel["completeness"] > 0.7 else ("partial" if rel["completeness"] > 0.3 else "incomplete"),
        "hallucination": "low" if hall["hallucination_score"] < 0.2 else ("medium" if hall["hallucination_score"] < 0.5 else "high"),
    }

    return report


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="LLM evaluation pipeline - Option A (single-file).")
    p.add_argument("--conv", required=True, help="Path to conversation JSON (sample-chat-conversation-01.json)")
    p.add_argument("--context", required=True, help="Path to context vectors JSON (sample_context_vector_01.json)")
    p.add_argument("--out", default="report.json", help="Output report JSON path")
    p.add_argument("--token-price-per-1k", type=float, default=0.002, help="USD price per 1k tokens (est)")
    p.add_argument("--hallucination-threshold", type=float, default=0.45, help="Threshold for context support (0..1)")
    return p.parse_args()


def main():
    args = parse_args()
    conv = load_json(args.conv)
    context = load_json(args.context)

    config = {
        "token_price_per_1k": args.token_price_per_1k,
        "hallucination_threshold": args.hallucination_threshold,
        "include_context_tokens": True,
    }

    report = evaluate(conv, context, config)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Evaluation complete. Report written to {args.out}")
    print(f"Aggregate score: {report.get('aggregate_score'):.3f}")
    print("Verdicts:", json.dumps(report.get("verdicts", {}), ensure_ascii=False))


if __name__ == "__main__":
    main()
