"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import os
import re
from pathlib import Path
from typing import Any

try:
    import chromadb
    import numpy as np
    from rank_bm25 import BM25Okapi
except ImportError:
    pass

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
DATA_STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
COLLECTION_NAME = "university_services_docs"

CORPUS: list[dict[str, Any]] = []  # List of {'content': str, 'metadata': dict}
_bm25 = None


def tokenize_text(text: str) -> list[str]:
    """Tách từ đơn giản cho BM25 (chuyển chữ thường, xóa ký tự đặc biệt)."""
    text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
    return [w for w in text_clean.split() if len(w) > 1]


def get_corpus() -> list[dict[str, Any]]:
    """
    Nạp corpus từ ChromaDB vector store.
    Nếu ChromaDB chưa khởi tạo, fallback đọc trực tiếp từ data/standardized/.
    """
    global CORPUS
    if CORPUS:
        return CORPUS

    # 1. Thử lấy từ ChromaDB
    try:
        if CHROMA_DIR.exists():
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            collection = client.get_collection(name=COLLECTION_NAME)
            results = collection.get(include=["documents", "metadatas"])
            if results and results.get("documents"):
                for doc, meta in zip(results["documents"], results["metadatas"]):
                    CORPUS.append({"content": doc, "metadata": meta or {}})
                print(f"[LexicalSearch] Đã nạp {len(CORPUS)} chunks từ ChromaDB.")
                return CORPUS
    except Exception as e:
        print(f"[LexicalSearch] Chưa nạp được từ ChromaDB ({e}). Đang fallback nạp file Markdown...")

    # 2. Fallback: Đọc từ data/standardized/
    if DATA_STANDARDIZED_DIR.exists():
        md_files = list(DATA_STANDARDIZED_DIR.glob("**/*.md"))
        for file_path in md_files:
            try:
                text = file_path.read_text(encoding="utf-8").strip()
                if not text:
                    continue
                # Split đơn giản theo đoạn văn
                paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
                for idx, p in enumerate(paragraphs):
                    CORPUS.append({
                        "content": p,
                        "metadata": {
                            "source": file_path.name,
                            "chunk_id": idx,
                            "category": "standardized"
                        }
                    })
            except Exception:
                continue
        print(f"[LexicalSearch] Fallback: Đã nạp {len(CORPUS)} chunks từ file Markdown.")

    return CORPUS


def build_bm25_index(corpus: list[dict[str, Any]]) -> Any:
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [tokenize_text(doc["content"]) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """
    Tìm kiếm từ khóa sử dụng BM25 (Sparse Retrieval).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    global _bm25
    corpus = get_corpus()
    if not corpus:
        return []

    if _bm25 is None:
        _bm25 = build_bm25_index(corpus)

    tokenized_query = tokenize_text(query)
    if not tokenized_query:
        return []

    scores = _bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score_val = float(scores[idx])
        if score_val > 0.0:
            results.append({
                "content": corpus[idx]["content"],
                "score": score_val,
                "metadata": corpus[idx]["metadata"]
            })
            
    return results


if __name__ == "__main__":
    # Test thử nghiệm module Lexical Search
    test_query = "tuition fee payment methods quy định học phí"
    print(f"🔍 Đang test Lexical Search cho query: '{test_query}'...")
    search_results = lexical_search(test_query, top_k=3)
    for idx, r in enumerate(search_results, 1):
        print(f" [{idx}] Score: {r['score']:.3f} | Source: {r['metadata'].get('source', 'N/A')}")
        print(f"     Content: {r['content'][:100]}...\n")
