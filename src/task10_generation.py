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


def format_context(chunks: list[dict]) -> str:
    """Format context với nhãn nguồn ổn định cho citation."""
    parts: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata") or {}
        source = str(metadata.get("source") or f"Source {index}")
        doc_type = str(metadata.get("type") or "unknown")
        section = metadata.get("section")
        label = f"Tài liệu {index} | Nguồn: {source} | Loại: {doc_type}"
        if section:
            label += f" | Mục: {section}"
        parts.append(f"[{label}]\n{str(chunk.get('content', '')).strip()}")
    return "\n\n---\n\n".join(parts)


def _source_summaries(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, Any]] = set()
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        key = (str(metadata.get("source", "unknown")), metadata.get("chunk_index"))
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "source": key[0],
            "chunk_index": key[1],
            "score": float(chunk.get("score", 0.0)),
            "content": chunk.get("content", ""),
            "metadata": metadata,
        })
    return sources


def _extractive_answer(chunks: list[dict[str, Any]]) -> str:
    """Zero-cost answer: trích đoạn liên quan nhất và gắn citation nguồn."""
    if not chunks:
        return NO_EVIDENCE_ANSWER
    parts: list[str] = []
    for chunk in chunks[:2]:
        metadata = chunk.get("metadata") or {}
        source = str(metadata.get("source") or "Nguồn hiện có")
        content = str(chunk.get("content", "")).strip()
        if not content:
            continue
        excerpt = content[:700].rsplit(" ", 1)[0].strip()
        parts.append(f"{excerpt} [{source}]")
    return "\n\n".join(parts) or NO_EVIDENCE_ANSWER


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict[str, Any]:
    """Retrieve, reorder và gọi OpenRouter; degrade an toàn khi thiếu evidence/API."""
    query = query.strip()
    if not query:
        return {"answer": NO_EVIDENCE_ANSWER, "sources": [], "retrieval_source": "none"}
    chunks = retrieve(query, top_k=top_k)
    retrieval_source = chunks[0].get("source", "hybrid") if chunks else "none"
    sources = _source_summaries(chunks)
    if not chunks:
        return {"answer": NO_EVIDENCE_ANSWER, "sources": sources, "retrieval_source": retrieval_source}

    if not ALLOW_EXTERNAL_APIS:
        return {
            "answer": _extractive_answer(chunks),
            "sources": sources,
            "retrieval_source": retrieval_source,
            "generation_mode": "local_extractive",
        }

    context = format_context(reorder_for_llm(chunks))
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return {"answer": NO_EVIDENCE_ANSWER, "sources": sources, "retrieval_source": retrieval_source}

    user_message = f"CONTEXT:\n{context}\n\n---\n\nCÂU HỎI: {query}"
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=800,
        )
        answer = (response.choices[0].message.content or "").strip() or NO_EVIDENCE_ANSWER
    except Exception as exc:
        answer = NO_EVIDENCE_ANSWER
        return {
            "answer": answer,
            "sources": sources,
            "retrieval_source": retrieval_source,
            "generation_error": f"{type(exc).__name__}: {exc}",
        }
    return {"answer": answer, "sources": sources, "retrieval_source": retrieval_source}


if __name__ == "__main__":
    result = generate_with_citation("Lương thử việc tối thiểu bằng bao nhiêu phần trăm?")
    print(result["answer"])
    print(f"Sources: {len(result['sources'])} via {result['retrieval_source']}")
