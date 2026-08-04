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

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests
import numpy as np
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

STANDARDIZED_DIR = Path(__file__).resolve().parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

# 800 ký tự giữ đủ ngữ cảnh cho điều/khoản pháp luật; overlap 100 giúp câu ở
# biên chunk không mất ngữ cảnh mà không tạo quá nhiều dữ liệu trùng.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"

# Voyage 4 Large hỗ trợ multilingual retrieval; cấu hình qua .env để có thể đổi
# model/dimension mà không sửa code hoặc lộ API key trong repository.
EMBEDDING_MODEL = os.getenv("VOYAGE_MODEL", "voyage-4-large")
EMBEDDING_DIM = int(os.getenv("VOYAGE_OUTPUT_DIMENSION", "1024"))
VOYAGE_BASE_URL = os.getenv("VOYAGE_BASE_URL", "https://api.voyageai.com/v1").rstrip("/")
VOYAGE_BATCH_SIZE = 64
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "labor_law_docs"
EMBEDDING_CACHE_PATH = PROJECT_DIR / "data" / "cache" / "voyage_embeddings.npz"


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


class VoyageEmbeddingClient:
    """REST client nhỏ gọn cho Voyage, có backoff với 429 và lỗi 5xx."""

    def __init__(self) -> None:
        api_key = os.getenv("VOYAGE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Thiếu VOYAGE_API_KEY trong file .env")
        self.endpoint = f"{VOYAGE_BASE_URL}/embeddings"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        retry = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"POST"}),
            respect_retry_after_header=True,
        )
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        """Embed một batch theo đúng mode ``document`` hoặc ``query``."""
        if input_type not in {"document", "query"}:
            raise ValueError("input_type phải là 'document' hoặc 'query'")
        if not texts:
            return []
        payload = {
            "input": texts,
            "model": EMBEDDING_MODEL,
            "input_type": input_type,
            "output_dimension": EMBEDDING_DIM,
            "output_dtype": "float",
            "truncation": True,
        }
        response = self.session.post(
            self.endpoint,
            headers=self.headers,
            json=payload,
            timeout=120,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text[:500]
            raise RuntimeError(f"Voyage API lỗi HTTP {response.status_code}: {detail}") from exc
        data = response.json().get("data", [])
        data.sort(key=lambda item: item.get("index", 0))
        embeddings = [item["embedding"] for item in data]
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Voyage trả {len(embeddings)} embedding cho {len(texts)} input"
            )
        for embedding in embeddings:
            if len(embedding) != EMBEDDING_DIM:
                raise RuntimeError(
                    f"Voyage embedding dimension {len(embedding)} != {EMBEDDING_DIM}"
                )
        return embeddings


@lru_cache(maxsize=1)
def get_embedding_client() -> VoyageEmbeddingClient:
    """Dùng chung HTTP session Voyage trong một process."""
    return VoyageEmbeddingClient()


def embed_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Gọi Voyage theo batch, dùng input_type=document cho corpus."""
    if not chunks:
        return chunks
    fingerprint_source = "\n".join(
        [EMBEDDING_MODEL, str(EMBEDDING_DIM), *(chunk["content"] for chunk in chunks)]
    )
    fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()
    if EMBEDDING_CACHE_PATH.exists():
        try:
            with np.load(EMBEDDING_CACHE_PATH, allow_pickle=False) as cached:
                cached_fingerprint = str(cached["fingerprint"].item())
                cached_embeddings = cached["embeddings"]
            expected_shape = (len(chunks), EMBEDDING_DIM)
            if cached_fingerprint == fingerprint and cached_embeddings.shape == expected_shape:
                for chunk, embedding in zip(chunks, cached_embeddings):
                    chunk["embedding"] = embedding.tolist()
                print(f"  Reused {len(chunks)} cached Voyage embeddings", flush=True)
                return chunks
        except (OSError, ValueError, KeyError):
            pass

    client = get_embedding_client()
    all_embeddings: list[list[float]] = []
    for start in range(0, len(chunks), VOYAGE_BATCH_SIZE):
        batch = chunks[start:start + VOYAGE_BATCH_SIZE]
        embeddings = client.embed(
            [chunk["content"] for chunk in batch], input_type="document"
        )
        all_embeddings.extend(embeddings)
        for chunk, embedding in zip(batch, embeddings):
            chunk["embedding"] = embedding
        completed = min(start + len(batch), len(chunks))
        print(f"  Embedded {completed}/{len(chunks)} chunks", flush=True)
    EMBEDDING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        EMBEDDING_CACHE_PATH,
        fingerprint=np.array(fingerprint),
        embeddings=np.asarray(all_embeddings, dtype=np.float32),
    )
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng Voyage API.

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))

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


def index_to_vectorstore(chunks: list[dict[str, Any]]) -> None:
    """Thay collection cũ bằng index mới để không trộn corpus giữa các lần chạy."""
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
            "embedding_provider": "voyage",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": EMBEDDING_DIM,
        },
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


def run_pipeline() -> None:
    """Chạy load → chunk → embed → index và in thống kê."""
    print("=" * 60)
    print("Task 4: Chunking & Indexing")
    print(f"Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"Embedding API: Voyage/{EMBEDDING_MODEL} ({EMBEDDING_DIM} dimensions)")
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
