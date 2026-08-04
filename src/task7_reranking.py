"""Task 7 — reranking bằng Reciprocal Rank Fusion (RRF, k=60)."""
from __future__ import annotations

import math
from typing import Any


def _identity(item: dict[str, Any]) -> str:
    """Định danh cùng chunk giữa nhiều ranker, fallback về content."""
    metadata = item.get("metadata") or {}
    source_path = metadata.get("source_path") or metadata.get("source")
    chunk_index = metadata.get("chunk_index")
    if source_path is not None and chunk_index is not None:
        return f"{source_path}::{chunk_index}"
    return str(item.get("content", ""))

import math
import re
from typing import Optional


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        length = min(len(a), len(b))
        a = a[:length]
        b = b[:length]
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _build_token_vector(text: str) -> list[float]:
    tokens = _tokenize(text)
    if not tokens:
        return []
    vector = {}
    for token in tokens:
        vector[token] = vector.get(token, 0) + 1.0
    return [vector.get(token, 0.0) for token in sorted(vector)]


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates bằng cách dùng một heuristic nhẹ dựa trên overlap từ khóa.
    Đây là fallback hữu ích khi không có cross-encoder model thực sự.
    """
    if not candidates:
        return []

    query_tokens = set(_tokenize(query))
    scored = []
    for candidate in candidates:
        content = candidate.get("content", "")
        content_tokens = set(_tokenize(content))
        overlap = len(query_tokens & content_tokens)
        base_score = float(candidate.get("score", 0.0))
        rerank_score = base_score + (overlap * 0.05)
        item = dict(candidate)
        item["score"] = rerank_score
        scored.append(item)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.
    """
    if not candidates:
        return []

    selected: list[dict] = []
    remaining = list(candidates)

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx, candidate in enumerate(remaining):
            emb = candidate.get("embedding")
            if emb is None:
                emb = _build_token_vector(candidate.get("content", ""))

            relevance = _cosine_similarity(query_embedding, emb)
            max_sim_to_selected = 0.0
            for selected_item in selected:
                selected_emb = selected_item.get("embedding")
                if selected_emb is None:
                    selected_emb = _build_token_vector(selected_item.get("content", ""))
                max_sim_to_selected = max(max_sim_to_selected, _cosine_similarity(emb, selected_emb))

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break

        selected.append(remaining.pop(best_idx))

    return selected


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """Gộp nhiều ranked list bằng ``sum(1 / (k + rank))``."""
    if top_k <= 0:
        return []
    if k < 0:
        raise ValueError("RRF k phải >= 0")
    scores: dict[str, float] = {}
    items: dict[str, dict[str, Any]] = {}
    ranks: dict[str, dict[str, int]] = {}
    for list_index, ranked_list in enumerate(ranked_lists):
        seen_in_list: set[str] = set()
        for rank, candidate in enumerate(ranked_list, 1):
            key = _identity(candidate)
            if not key or key in seen_in_list:
                continue
            seen_in_list.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            items.setdefault(key, candidate.copy())
            ranks.setdefault(key, {})[f"ranker_{list_index}"] = rank
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    results: list[dict[str, Any]] = []
    for key in ordered[:top_k]:
        item = items[key].copy()
        item["score"] = float(scores[key])
        item["rrf_ranks"] = ranks[key]
        results.append(item)
    return results

    RRF(d) = Σ 1 / (k + rank_r(d))
    """
    if not ranked_lists:
        return []

    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item.get("content", "")
            if not key:
                continue
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = dict(item)

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = round(score, 6)
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",
) -> list[dict]:
    """
    Unified reranking interface.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # Cần query_embedding - embed query trước
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    elif method == "rrf":
        if candidates and isinstance(candidates[0], list):
            return rerank_rrf(candidates, top_k=top_k)
        return rerank_cross_encoder(query, candidates, top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dense = [
        {"content": "Lương thử việc", "score": 0.8, "metadata": {"chunk_index": 1}},
        {"content": "Nghỉ phép năm", "score": 0.7, "metadata": {"chunk_index": 2}},
    ]
    sparse = list(reversed(dense))
    for result in rerank_rrf([dense, sparse]):
        print(f"[{result['score']:.5f}] {result['content']}")
