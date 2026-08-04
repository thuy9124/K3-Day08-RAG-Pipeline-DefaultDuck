"""Task 5 — dense retrieval local từ ChromaDB."""
from __future__ import annotations

from typing import Any

from .task4_chunking_indexing import get_collection, get_embedding_client


def semantic_search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Trả tối đa ``top_k`` chunk theo cosine similarity giảm dần."""
    query = query.strip()
    if not query or top_k <= 0:
        return []
    try:
        collection = get_collection()
    except (ValueError, KeyError):
        return []
    count = collection.count()
    if count == 0:
        return []
    query_vector = get_embedding_client().embed([query], input_type="query")[0]
    raw = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )
    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    results = [{
        "content": document,
        "score": float(max(-1.0, min(1.0, 1.0 - distance))),
        "metadata": metadata or {},
    } for document, metadata, distance in zip(documents, metadatas, distances)]
    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    for result in semantic_search("Lương thử việc tối thiểu là bao nhiêu?", top_k=5):
        print(f"[{result['score']:.3f}] {result['metadata'].get('source')}")
