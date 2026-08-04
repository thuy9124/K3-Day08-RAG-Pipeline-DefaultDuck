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


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(value * value for value in left))
    norm_right = math.sqrt(sum(value * value for value in right))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0


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
    """Không dùng trong cấu hình hiện tại; giữ lỗi rõ thay vì âm thầm giả score."""
    raise RuntimeError("Cross-encoder chưa được cấu hình; pipeline đang sử dụng RRF")


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Chọn candidates cân bằng relevance/diversity bằng MMR."""
    if top_k <= 0 or not candidates:
        return []
    if not 0 <= lambda_param <= 1:
        raise ValueError("lambda_param phải nằm trong [0, 1]")
    selected: list[int] = []
    remaining = list(range(len(candidates)))
    while remaining and len(selected) < top_k:
        best_index = remaining[0]
        best_score = float("-inf")
        for index in remaining:
            embedding = candidates[index].get("embedding", [])
            relevance = _cosine(query_embedding, embedding)
            redundancy = max(
                (_cosine(embedding, candidates[chosen].get("embedding", [])) for chosen in selected),
                default=0.0,
            )
            score = lambda_param * relevance - (1 - lambda_param) * redundancy
            if score > best_score:
                best_index, best_score = index, score
        item = candidates[best_index].copy()
        item["score"] = float(best_score)
        selected.append(best_index)
        remaining.remove(best_index)
        candidates[best_index] = item
    return [candidates[index] for index in selected]


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


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",
) -> list[dict]:
    """Unified interface; một candidate list được coi là một ranker với RRF."""
    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        raise ValueError("MMR cần query_embedding; hãy gọi rerank_mmr trực tiếp")
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dense = [
        {"content": "Lương thử việc", "score": 0.8, "metadata": {"chunk_index": 1}},
        {"content": "Nghỉ phép năm", "score": 0.7, "metadata": {"chunk_index": 2}},
    ]
    sparse = list(reversed(dense))
    for result in rerank_rrf([dense, sparse]):
        print(f"[{result['score']:.5f}] {result['content']}")
