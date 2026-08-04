"""Task 4 — chunk Markdown, tạo embedding local và index vào ChromaDB.

Backend mặc định dùng HashingVectorizer của scikit-learn: không tải model, không gọi
API và không phát sinh chi phí. Character n-gram phù hợp với tiếng Việt và cho cùng
một không gian vector ổn định giữa lúc index và lúc query.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

STANDARDIZED_DIR = Path(__file__).resolve().parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"
EMBEDDING_MODEL = "local-char-ngram-hashing-v1"
EMBEDDING_DIM = 2048
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "labor_law_docs"


def load_documents() -> list[dict[str, Any]]:
    """Đọc Markdown không rỗng, kèm metadata nguồn và loại tài liệu."""
    documents: list[dict[str, Any]] = []
    if not STANDARDIZED_DIR.exists():
        return documents
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8-sig", errors="replace").strip()
        if not content:
            continue
        relative = md_file.relative_to(STANDARDIZED_DIR)
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "source_path": relative.as_posix(),
                "type": relative.parts[0] if len(relative.parts) > 1 else "unknown",
            },
        })
    return documents


def chunk_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cắt tài liệu bằng RecursiveCharacterTextSplitter và giữ metadata nguồn."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", "; ", " ", ""],
    )
    chunks: list[dict[str, Any]] = []
    for document in documents:
        text = str(document.get("content", "")).strip()
        if not text:
            continue
        for index, chunk_text in enumerate(splitter.split_text(text)):
            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append({
                    "content": chunk_text,
                    "metadata": {**document.get("metadata", {}), "chunk_index": index},
                })
    return chunks


class LocalEmbeddingClient:
    """Embedding local, deterministic; interface tương thích document/query."""

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self.vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            n_features=EMBEDDING_DIM,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
        )

    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        if input_type not in {"document", "query"}:
            raise ValueError("input_type phải là 'document' hoặc 'query'")
        if not texts:
            return []
        return self.vectorizer.transform(texts).astype("float32").toarray().tolist()


@lru_cache(maxsize=1)
def get_embedding_client() -> LocalEmbeddingClient:
    return LocalEmbeddingClient()


def embed_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tạo embedding local theo batch để giới hạn bộ nhớ."""
    client = get_embedding_client()
    batch_size = 256
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        embeddings = client.embed([item["content"] for item in batch], "document")
        for chunk, embedding in zip(batch, embeddings):
            chunk["embedding"] = embedding
    return chunks


def get_chroma_client():
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    return get_chroma_client().get_collection(name=COLLECTION_NAME)


def index_to_vectorstore(chunks: list[dict[str, Any]]) -> None:
    """Thay collection cũ bằng index mới để không trộn nhiều phiên bản corpus."""
    from chromadb.errors import NotFoundError

    if not chunks:
        raise ValueError("Không có chunk để index")
    if any("embedding" not in chunk for chunk in chunks):
        raise ValueError("Các chunk phải được embed trước khi index")
    client = get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except (ValueError, NotFoundError):
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_provider": "local",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": EMBEDDING_DIM,
        },
    )
    for start in range(0, len(chunks), 256):
        batch = chunks[start:start + 256]
        ids = []
        for chunk in batch:
            meta = chunk["metadata"]
            raw_id = f"{meta.get('source_path')}:{meta.get('chunk_index')}"
            ids.append(hashlib.sha1(raw_id.encode("utf-8")).hexdigest())
        collection.add(
            ids=ids,
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=[chunk["metadata"] for chunk in batch],
        )


def run_pipeline() -> None:
    print("=" * 60)
    print("Task 4: Chunking & Indexing — zero-cost local mode")
    print(f"Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"Embedding: {EMBEDDING_MODEL} ({EMBEDDING_DIM} dimensions)")
    print(f"Vector store: {VECTOR_STORE} -> {CHROMA_DIR}")
    print("=" * 60)
    documents = load_documents()
    if not documents:
        raise RuntimeError(f"Không có Markdown hợp lệ trong {STANDARDIZED_DIR}")
    chunks = chunk_documents(documents)
    embed_chunks(chunks)
    index_to_vectorstore(chunks)
    print(f"Loaded {len(documents)} documents; indexed {get_collection().count()} chunks")


if __name__ == "__main__":
    run_pipeline()
