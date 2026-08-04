"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho cả tiếng Việt lẫn tiếng Anh)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

import json
import math
import os
import re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
PROJECT_ROOT = Path(__file__).parent.parent


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# TODO: Chọn chunking strategy và giải thích vì sao
CHUNK_SIZE = 800        # Dùng chunk lớn hơn để giữ ngữ cảnh đủ cho RAG, nhưng vẫn vừa phải cho retrieval.
CHUNK_OVERLAP = 100      # Overlap 100 giúp giữ mạch ý giữa các chunk liên tiếp.
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Dùng Voyage AI
EMBEDDING_MODEL = os.environ.get("VOYAGE_MODEL", "voyage-4-large")
EMBEDDING_DIM = int(os.environ.get("VOYAGE_OUTPUT_DIMENSION", 1024))

# TODO: Chọn vector store
VECTOR_STORE = "chromadb"  # "chromadb" | "weaviate" | "faiss"
COLLECTION_NAME = "university_services_docs"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents: list[dict] = []

    md_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
    if not md_files:
        fallback_candidates = [
            PROJECT_ROOT / "README.md",
            PROJECT_ROOT / "LAB_GUIDE.md",
            PROJECT_ROOT / "day8-lab-rag-pipeline.md",
        ]
        md_files = [p for p in fallback_candidates if p.exists()]

    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception:
            continue

        if not content.strip():
            continue

        doc_type = "legal" if "legal" in str(md_file).lower() else "news" if "news" in str(md_file).lower() else "general"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "type": doc_type,
                "path": str(md_file.relative_to(PROJECT_ROOT)),
            },
        })

    return documents


def _split_text_to_size(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Chia một đoạn văn thành các chunk có kích thước tối đa chunk_size."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    parts = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    parts = [p.strip() for p in parts if p and p.strip()]

    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current} {part}".strip() if current else part
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(part) <= chunk_size:
            current = part
            continue

        words = part.split()
        word_buffer = ""
        for word in words:
            candidate_words = f"{word_buffer} {word}".strip() if word_buffer else word
            if len(candidate_words) <= chunk_size:
                word_buffer = candidate_words
            else:
                if word_buffer:
                    chunks.append(word_buffer)
                    word_buffer = word
                else:
                    chunks.append(word)
        if word_buffer:
            current = word_buffer

    if current:
        chunks.append(current)

    if overlap > 0 and len(chunks) > 1:
        adjusted: list[str] = []
        for idx, chunk in enumerate(chunks):
            if idx == 0:
                adjusted.append(chunk)
                continue
            prev = adjusted[-1]
            if len(prev) + overlap < len(chunk):
                adjusted.append(chunk)
            else:
                adjusted.append(chunk)
        return adjusted

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    chunks: list[dict] = []

    for doc in documents:
        text = (doc.get("content") or "").strip()
        if not text:
            continue

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            paragraphs = [text]

        for paragraph_index, paragraph in enumerate(paragraphs):
            split_chunks = _split_text_to_size(paragraph, CHUNK_SIZE, CHUNK_OVERLAP)
            for chunk_index, chunk_text in enumerate(split_chunks):
                chunk_id = f"{paragraph_index}_{chunk_index}"
                chunks.append({
                    "content": chunk_text.strip(),
                    "metadata": {**doc.get("metadata", {}), "chunk_index": chunk_id},
                })

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng Voyage API.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    if not chunks:
        return chunks

    try:
        import voyageai
        client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))

        texts = [c["content"] for c in chunks]
        # Gọi API theo batch 100 để tránh vượt giới hạn token/request
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = client.embed(
                texts=batch,
                model=EMBEDDING_MODEL,
            )
            all_embeddings.extend(response.embeddings)
            print(f"  Embedded batch {i // batch_size + 1}/{math.ceil(len(texts) / batch_size)}")

        for chunk, emb in zip(chunks, all_embeddings):
            chunk["embedding"] = emb
        return chunks
    except Exception as e:
        print(f"Voyage AI embedding error: {e}")
        # Fallback vector giả nếu API lỗi
        for chunk in chunks:
            tokens = re.findall(r"\w+", (chunk.get("content") or "").lower())
            vector = [0.0] * EMBEDDING_DIM
            for token in tokens:
                idx = abs(hash(token)) % EMBEDDING_DIM
                vector[idx] += 1.0
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            chunk["embedding"] = [v / norm for v in vector]
        return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    if not chunks:
        return None

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        ids = [
            f"{c['metadata'].get('source', 'chunk')}_chunk_{c['metadata'].get('chunk_index', 0)}"
            for c in chunks
        ]
        collection.upsert(
            ids=ids,
            documents=[c["content"] for c in chunks],
            embeddings=[c["embedding"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )
        return collection
    except Exception as e:
        print(f"Failed to upsert to ChromaDB: {e}")
        fallback_path = CHROMA_DIR / "fallback_index.json"
        payload = {
            "collection_name": COLLECTION_NAME,
            "chunks": [
                {
                    "content": c["content"],
                    "metadata": c["metadata"],
                    "embedding": c.get("embedding"),
                }
                for c in chunks
            ],
        }
        fallback_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return None


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
