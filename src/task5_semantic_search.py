"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import os
import chromadb
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "university_services_docs"
# Dùng Voyage AI
EMBEDDING_MODEL = os.environ.get("VOYAGE_MODEL", "voyage-4-large")

_client = None
_collection = None


def _embed_query(query: str) -> list[float]:
    """Embed query bằng Voyage AI."""
    import voyageai
    client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))
    response = client.embed(texts=[query], model=EMBEDDING_MODEL)
    return response.embeddings[0]


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
    return _collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score [0, 1]
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    query_vector = _embed_query(query)

    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    if results["documents"] and len(results["documents"]) > 0:
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            score = max(0.0, 1.0 - dist)  # cosine distance → similarity
            output.append({"content": doc, "score": round(score, 4), "metadata": meta})

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    for result in semantic_search("Lương thử việc tối thiểu là bao nhiêu?", top_k=5):
        print(f"[{result['score']:.3f}] {result['metadata'].get('source')}")
        print(result["content"][:180], "\n")
