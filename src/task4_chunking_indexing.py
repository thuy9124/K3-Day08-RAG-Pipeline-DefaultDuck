"""Task 4 — chunk Markdown, embed bằng BGE-M3 và index vào ChromaDB."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

STANDARDIZED_DIR = Path(__file__).resolve().parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

# 800 ký tự giữ đủ ngữ cảnh cho điều/khoản pháp luật; overlap 100 giúp câu ở
# biên chunk không mất ngữ cảnh mà không tạo quá nhiều dữ liệu trùng.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"

# BGE-M3 là model multilingual 1024 chiều, phù hợp corpus pháp luật tiếng Việt.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "labor_law_docs"


def load_documents() -> list[dict[str, Any]]:
    """Đọc các Markdown không rỗng, kèm metadata nguồn và loại tài liệu."""
    documents: list[dict[str, Any]] = []
    if not STANDARDIZED_DIR.exists():
        return documents
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8-sig", errors="replace").strip()
        if not content:
            continue
        relative = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative.parts[0] if len(relative.parts) > 1 else "unknown"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "source_path": relative.as_posix(),
                "type": doc_type,
            },
        })
    return documents


def chunk_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cắt tài liệu bằng RecursiveCharacterTextSplitter, giữ metadata nguồn."""
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
            if not chunk_text:
                continue
            chunks.append({
                "content": chunk_text,
                "metadata": {**document.get("metadata", {}), "chunk_index": index},
            })
    return chunks


@lru_cache(maxsize=1)
def get_embedding_model():
    """Lazy-load một instance SentenceTransformer dùng chung toàn pipeline."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Thêm normalized embedding 1024 chiều vào từng chunk."""
    if not chunks:
        return chunks
    embeddings = get_embedding_model().encode(
        [chunk["content"] for chunk in chunks],
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    if embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension không đúng: {embeddings.shape[1]} != {EMBEDDING_DIM}"
        )
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()
    return chunks


def get_chroma_client():
    """Mở Chroma persistent client tại thư mục project."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    """Mở collection cosine đã index; không tự tạo collection rỗng."""
    return get_chroma_client().get_collection(name=COLLECTION_NAME)


def index_to_vectorstore(chunks: list[dict[str, Any]]) -> None:
    """Thay collection cũ bằng index mới để không trộn corpus giữa các lần chạy."""
    if not chunks:
        raise ValueError("Không có chunk để index")
    if any("embedding" not in chunk for chunk in chunks):
        raise ValueError("Các chunk phải được embed trước khi index")

    client = get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except ValueError:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
    )
    batch_size = 256
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
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
    """Chạy load → chunk → embed → index và in thống kê."""
    print("=" * 60)
    print("Task 4: Chunking & Indexing")
    print(f"Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"Embedding: {EMBEDDING_MODEL} ({EMBEDDING_DIM} dimensions)")
    print(f"Vector store: {VECTOR_STORE} -> {CHROMA_DIR}")
    print("=" * 60)
    documents = load_documents()
    if not documents:
        raise RuntimeError(f"Không có Markdown hợp lệ trong {STANDARDIZED_DIR}")
    print(f"Loaded {len(documents)} documents")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")
    embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks")
    index_to_vectorstore(chunks)
    print(f"Indexed {get_collection().count()} chunks")


if __name__ == "__main__":
    run_pipeline()
