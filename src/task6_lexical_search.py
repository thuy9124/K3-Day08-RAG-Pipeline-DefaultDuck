"""Task 6 — sparse lexical retrieval với BM25 cho văn bản tiếng Việt."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from .task4_chunking_indexing import chunk_documents, load_documents

TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize Unicode đơn giản, giữ từ có dấu và số điều/khoản."""
    return TOKEN_PATTERN.findall(text.casefold())


def build_bm25_index(corpus: list[dict[str, Any]]) -> BM25Okapi:
    """Xây BM25Okapi từ corpus chunk; corpus rỗng là lỗi cấu hình rõ ràng."""
    if not corpus:
        raise ValueError("Corpus BM25 rỗng")
    return BM25Okapi([tokenize(document["content"]) for document in corpus], k1=1.5, b=0.75)


@lru_cache(maxsize=1)
def _get_index() -> tuple[list[dict[str, Any]], BM25Okapi]:
    corpus = chunk_documents(load_documents())
    return corpus, build_bm25_index(corpus)


def refresh_index() -> None:
    """Xóa cache BM25 sau khi corpus standardized thay đổi."""
    _get_index.cache_clear()


def lexical_search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Trả các chunk có BM25 score dương, sắp xếp giảm dần."""
    tokens = tokenize(query)
    if not tokens or top_k <= 0:
        return []
    try:
        corpus, bm25 = _get_index()
    except ValueError:
        return []
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = [
        {
            "content": corpus[int(index)]["content"],
            "score": float(scores[index]),
            "metadata": corpus[int(index)]["metadata"],
        }
        for index in top_indices
        if scores[index] > 0
    ]
    return results


if __name__ == "__main__":
    for result in lexical_search("lương thử việc", top_k=5):
        print(f"[{result['score']:.3f}] {result['metadata'].get('source')}")
        print(result["content"][:180], "\n")
