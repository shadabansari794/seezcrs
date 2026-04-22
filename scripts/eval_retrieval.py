"""Offline retrieval evaluation against LLM-Redial ground-truth rec_items.

Runs retrieval-only (no LLM calls) to measure how well the vector store
surfaces the ground-truth recommendation given a query built from the
user's conversation likes/dislikes.

Metrics:
  Hit@K  — Was the ground-truth title found in the top K retrieved results?
  MRR    — Mean Reciprocal Rank of the ground-truth title across all queries.

Usage (inside Docker):
  docker compose exec movie-crs-api python scripts/eval_retrieval.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/llm_redial")
ITEM_MAP_PATH = DATA_DIR / "item_map.json"
FINAL_DATA_PATH = DATA_DIR / "final_data.jsonl"
RESULTS_PATH = Path("eval_retrieval_results.txt")

# ── Config ─────────────────────────────────────────────────────────────────
TOP_K_VALUES = [1, 3, 5, 10]
MAX_RETRIEVE = max(TOP_K_VALUES) * 2  # over-fetch for reranker headroom
MAX_CONVERSATIONS = 200  # cap for speed; set None for all


def load_item_map() -> Dict[str, str]:
    with open(ITEM_MAP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_conversations() -> List[Tuple[str, Dict[str, Any]]]:
    """Return flat list of (user_id, conversation_inner_dict) tuples."""
    conversations: List[Tuple[str, Dict[str, Any]]] = []
    with open(FINAL_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            for user_id, payload in record.items():
                for conv_entry in payload.get("Conversation", []):
                    for _key, inner in conv_entry.items():
                        conversations.append((user_id, inner))
    return conversations


def build_query(
    inner: Dict[str, Any], item_map: Dict[str, str]
) -> Optional[str]:
    """Build a natural-language query from the conversation's likes/dislikes."""
    likes = [item_map.get(str(iid)) for iid in inner.get("user_likes", [])]
    dislikes = [item_map.get(str(iid)) for iid in inner.get("user_dislikes", [])]
    likes = [t for t in likes if t]
    dislikes = [t for t in dislikes if t]

    if not likes and not dislikes:
        return None

    parts = []
    if likes:
        parts.append(f"I enjoyed {', '.join(likes)}.")
    if dislikes:
        parts.append(f"I did not like {', '.join(dislikes)}.")
    parts.append("Recommend me something similar.")
    return " ".join(parts)


def get_ground_truth_titles(
    inner: Dict[str, Any], item_map: Dict[str, str]
) -> List[str]:
    """Resolve rec_item ASINs to titles."""
    return [
        item_map[str(iid)]
        for iid in inner.get("rec_item", [])
        if str(iid) in item_map
    ]


def normalize(title: str) -> str:
    """Lowercase, strip whitespace for fuzzy matching."""
    return title.strip().lower()


def run_eval():
    print("=" * 60)
    print("LLM-Redial Retrieval Evaluation")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    item_map = load_item_map()
    print(f"Loaded {len(item_map)} items from item_map.json")

    all_convs = load_conversations()
    print(f"Loaded {len(all_convs)} total conversations from final_data.jsonl")

    # Filter to conversations that have both query material and ground truth
    valid_convs: List[Tuple[str, Dict[str, Any], str, List[str]]] = []
    for user_id, inner in all_convs:
        query = build_query(inner, item_map)
        gt_titles = get_ground_truth_titles(inner, item_map)
        if query and gt_titles:
            valid_convs.append((user_id, inner, query, gt_titles))

    if MAX_CONVERSATIONS and len(valid_convs) > MAX_CONVERSATIONS:
        valid_convs = valid_convs[:MAX_CONVERSATIONS]

    print(f"Evaluating {len(valid_convs)} conversations (with query + ground truth)")
    print()

    # ── Initialize vector store ────────────────────────────────────────────
    from app.config import settings
    from utils.vector_store import MovieVectorStore

    vs = MovieVectorStore(
        embedding_model=settings.embedding_model,
        persist_directory=settings.vector_store_path,
    )
    stats = vs.get_collection_stats()
    print(f"Vector store: {stats['total_movies']} movies indexed")

    # ── Optionally load reranker ───────────────────────────────────────────
    reranker = None
    try:
        from utils.reranker import MovieReranker
        reranker = MovieReranker()
        print("Reranker loaded: BAAI/bge-reranker-base")
    except Exception as e:
        print(f"Reranker not available ({e}), evaluating retrieval only")
    print()

    # ── Run evaluation ─────────────────────────────────────────────────────
    hits: Dict[int, int] = {k: 0 for k in TOP_K_VALUES}
    reciprocal_ranks: List[float] = []
    total = 0
    t_start = time.time()

    for idx, (user_id, inner, query, gt_titles) in enumerate(valid_convs):
        gt_set = {normalize(t) for t in gt_titles}

        # Retrieve candidates
        candidates = vs.search(query, top_k=MAX_RETRIEVE)

        # Rerank if available
        if reranker and candidates:
            candidates = reranker.rerank(query, candidates, top_k=max(TOP_K_VALUES))

        # Find rank of ground-truth item
        rank = None
        for i, movie in enumerate(candidates, start=1):
            if normalize(movie.get("title", "")) in gt_set:
                rank = i
                break

        # Update metrics
        for k in TOP_K_VALUES:
            if rank is not None and rank <= k:
                hits[k] += 1

        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        total += 1

        if (idx + 1) % 25 == 0:
            elapsed = time.time() - t_start
            print(f"  [{idx+1}/{len(valid_convs)}]  {elapsed:.1f}s elapsed")

    elapsed = time.time() - t_start

    # ── Compute metrics ────────────────────────────────────────────────────
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("LLM-Redial Retrieval Evaluation Results")
    lines.append("=" * 60)
    lines.append(f"Conversations evaluated: {total}")
    lines.append(f"Vector store size:       {stats['total_movies']} movies")
    lines.append(f"Embedding model:         {settings.embedding_model}")
    lines.append(f"Reranker:                {'BAAI/bge-reranker-base' if reranker else 'none'}")
    lines.append(f"Retrieval over-fetch:    top_{MAX_RETRIEVE}")
    lines.append(f"Time elapsed:            {elapsed:.1f}s")
    lines.append("")

    lines.append("Hit@K Metrics")
    lines.append("-" * 30)
    for k in TOP_K_VALUES:
        rate = hits[k] / total * 100 if total else 0
        lines.append(f"  Hit@{k:<3d}  {hits[k]:>4d}/{total}  ({rate:.1f}%)")

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
    lines.append("")
    lines.append(f"MRR (Mean Reciprocal Rank): {mrr:.4f}")
    lines.append("")

    # Breakdown: how many had rank=None (not found in top MAX_RETRIEVE at all)
    not_found = sum(1 for rr in reciprocal_ranks if rr == 0.0)
    lines.append(f"Ground truth not in top {MAX_RETRIEVE}: {not_found}/{total} ({not_found/total*100:.1f}%)")

    result_text = "\n".join(lines)
    print()
    print(result_text)

    # Save to file
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(result_text + "\n")
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_eval()
