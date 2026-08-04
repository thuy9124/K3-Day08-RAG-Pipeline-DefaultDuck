"""Task 9 — Hybrid Voyage dense + BM25 + RRF với vectorless fallback."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

SCORE_THRESHOLD = 0.48
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"
ALLOW_EXTERNAL_APIS = os.getenv("RAG_ALLOW_EXTERNAL_APIS", "false").casefold() in {
    "1", "true", "yes", "on"
}


def _deduplicate(results: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for item in results:
        metadata = item.get("metadata") or {}
        key = f"{metadata.get('source_path', metadata.get('source'))}:{metadata.get('chunk_index')}"
        if key in seen:
            continue
        seen.add(key)
        output.append(item.copy())
        if len(output) >= top_k:
            break
    return output


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict[str, Any]]:
    """Retrieve và dùng cosine dense gốc—not RRF score—để quyết định fallback."""
    query = query.strip()
    if not query or top_k <= 0:
        return []
    candidate_k = max(top_k * 2, top_k)
    if ALLOW_EXTERNAL_APIS:
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(semantic_search, query, candidate_k)
            sparse_future = executor.submit(lexical_search, query, candidate_k)
            try:
                dense_results = dense_future.result()
            except Exception:
                dense_results = []
            sparse_results = sparse_future.result()
    else:
        # Zero-cost mode: không gọi Voyage query API; BM25 và structural fallback
        # đều chạy local.
        dense_results = []
        sparse_results = lexical_search(query, candidate_k)

    best_dense_score = dense_results[0]["score"] if dense_results else 0.0
    if not ALLOW_EXTERNAL_APIS or best_dense_score < score_threshold:
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback

    if use_reranking:
        merged = rerank_rrf([dense_results, sparse_results], top_k=top_k)
    else:
        merged = _deduplicate(dense_results + sparse_results, top_k)
    for item in merged:
        item["source"] = "hybrid"
        item["dense_best_score"] = float(best_dense_score)
    return merged[:top_k]


if __name__ == "__main__":
    for query in ("lương thử việc tối thiểu", "xyzabc123nonsense"):
        print(f"\nQuery: {query}")
        for result in retrieve(query, top_k=3):
            print(f"[{result['score']:.4f}] [{result['source']}] {result['content'][:100]}")
