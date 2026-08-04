"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    # TODO: Implement upload
    #
    # Tham khảo: https://github.com/VectifyAI/PageIndex
    #
    # from pageindex.client import PageIndexClient
    #
    # client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    #
    # for md_file in STANDARDIZED_DIR.rglob("*.md"):
    #     # Lưu ý: PageIndex nhận PDF, không nhận .md trực tiếp — có thể cần
    #     # convert markdown sang PDF đơn giản bằng fpdf2 trước khi upload.
    #     resp = client.submit_document(str(pdf_path))
    #     doc_id = resp.get("doc_id") or resp.get("id")
    #     print(f"  ✓ Uploaded: {md_file.name} -> {doc_id}")
    raise NotImplementedError("Implement upload_documents")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not PAGEINDEX_API_KEY:
        # Fallback to dummy data for testing without API key
        return [{"content": "Dummy fallback content from PageIndex", "score": 0.5, "metadata": {"section": "Dummy Section"}, "source": "pageindex"}]
        
    try:
        from pageindex.client import PageIndexClient
        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
        resp = client.submit_query(query=query) 
        retrieval_id = resp.get("retrieval_id") or resp.get("id")
        
        # Poll cho đến khi status == "completed" (giả lập)
        retrieval = client.get_retrieval(retrieval_id)
        
        # Parse retrieval["retrieved_nodes"]
        results = []
        for node in retrieval.get("retrieved_nodes", [])[:2]:
            for group in node.get("relevant_contents", []):
                for item in group:
                    results.append({
                        "content": item.get("relevant_content", ""),
                        "score": 0.5,  # PageIndex không trả score trực tiếp — tự gán theo rank
                        "metadata": {"section": item.get("section_title")},
                        "source": "pageindex",
                    })
        return results[:top_k]
    except Exception as e:
        print(f"PageIndex API Error: {e}")
        return []


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("tuition fee payment methods", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
