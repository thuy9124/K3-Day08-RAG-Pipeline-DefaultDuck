"""Task 10 — grounded generation có citation qua OpenRouter."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .task9_retrieval_pipeline import ALLOW_EXTERNAL_APIS, retrieve

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
LLM_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
NO_EVIDENCE_ANSWER = "Tôi không thể xác minh thông tin này từ nguồn hiện có."

SYSTEM_PROMPT = """Bạn là trợ lý hỏi đáp pháp luật lao động Việt Nam cho người trẻ.

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin trong CONTEXT; không bổ sung kiến thức bên ngoài.
2. Mỗi khẳng định pháp lý phải có citation dạng [Tên nguồn].
3. Nếu context không đủ, trả lời đúng câu: "Tôi không thể xác minh thông tin này từ nguồn hiện có."
4. Trả lời bằng tiếng Việt, rõ ràng, trực tiếp và phân biệt quy định với ví dụ.
5. Không bịa số điều, ngày tháng, mức tiền hoặc quyền lợi."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đặt chunk quan trọng ở đầu/cuối để giảm lost-in-the-middle."""
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks
        
    front = chunks[::2]   # index 0, 2, 4 -> đặt ở đầu
    back = chunks[1::2]   # index 1, 3    -> đặt ở cuối (reversed)
    return front + back[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Source {i}")
        doc_type = chunk.get("metadata", {}).get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)
    
    # Step 2: Reorder
    reordered = reorder_for_llm(chunks)
    
    # Step 3: Format context
    context = format_context(reordered)
    
    # Step 4: Build prompt
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
    
    # Step 5: Call LLM (OpenRouter — OpenAI-compatible API)
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "answer": "Missing API key for generation.",
            "sources": chunks,
            "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none"
        }
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"Error calling LLM: {e}"
        
    # Step 6: Return
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none"
    }


if __name__ == "__main__":
    result = generate_with_citation("Lương thử việc tối thiểu bằng bao nhiêu phần trăm?")
    print(result["answer"])
    print(f"Sources: {len(result['sources'])} via {result['retrieval_source']}")
