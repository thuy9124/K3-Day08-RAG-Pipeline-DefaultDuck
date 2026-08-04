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
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → dense_results (giữ điểm cosine gốc)
          ├→ Lexical Search  → sparse_results
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If dense_results[0]["score"] < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm cosine gốc tối thiểu (KHÔNG phải điểm RRF)
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Song song chạy semantic + lexical
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)
    
    # Step 2: Merge bằng RRF
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"
        
    # Step 3: Rerank
    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        final_results = merged[:top_k]
        
    # Step 4: Check threshold DÙNG ĐIỂM COSINE GỐC (dense_results)
    best_score = dense_results[0]["score"] if dense_results else 0.0
    if best_score < score_threshold:
        print(f"  ⚠ Semantic best score ({best_score:.3f}) < threshold ({score_threshold})")
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback
            
    return final_results[:top_k]


if __name__ == "__main__":
    for query in ("lương thử việc tối thiểu", "xyzabc123nonsense"):
        print(f"\nQuery: {query}")
        for result in retrieve(query, top_k=3):
            print(f"[{result['score']:.4f}] [{result['source']}] {result['content'][:100]}")
