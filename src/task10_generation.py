"""Task 10 — grounded generation có citation qua OpenRouter."""
from __future__ import annotations

import os
import re
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


_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_STOPWORDS = {
    "và", "là", "có", "được", "cho", "của", "thì", "bao", "nhiêu", "một",
    "những", "các", "với", "trong", "tại", "về", "theo", "hay", "không",
    "người", "công", "ty", "lao", "động", "việc", "hướng", "dẫn", "quy", "định",
}
_OUT_OF_DOMAIN_PATTERN = re.compile(
    r"vũ\s*khí|hạt\s*nhân|du\s*hành\s*thời\s*gian|thế\s*kỷ\s*22|"
    r"động\s*cơ\s*phản\s*lực|siêu\s*thanh",
    re.IGNORECASE,
)


def _keywords(text: str) -> set[str]:
    return {
        token for token in _TOKEN_PATTERN.findall(text.casefold())
        if len(token) > 2 and token not in _STOPWORDS
    }


def _has_sufficient_evidence(query: str, chunks: list[dict[str, Any]]) -> bool:
    """Chặn câu hỏi ngoài domain khi các khái niệm chính không có trong evidence."""
    if _OUT_OF_DOMAIN_PATTERN.search(query):
        return False
    query_terms = _keywords(query)
    if not query_terms or not chunks:
        return False
    context_terms = _keywords(" ".join(str(item.get("content", "")) for item in chunks[:5]))
    overlap = query_terms & context_terms
    return len(overlap) >= 2 and len(overlap) / len(query_terms) >= 0.50


def _extractive_answer(query: str, chunks: list[dict[str, Any]]) -> str:
    """Chọn các câu bám sát query nhất và gắn citation nguồn, hoàn toàn local."""
    if not chunks:
        return NO_EVIDENCE_ANSWER
    query_terms = _keywords(query)
    candidates: list[tuple[int, int, str, str]] = []
    for chunk_index, chunk in enumerate(chunks[:5]):
        metadata = chunk.get("metadata") or {}
        source = str(metadata.get("source") or "Nguồn hiện có")
        content = str(chunk.get("content", "")).strip()
        for sentence in re.split(r"(?<=[.!?;:])\s+|\n+", content):
            sentence = sentence.strip()
            if len(sentence) < 35:
                continue
            score = len(query_terms & _keywords(sentence))
            if score:
                candidates.append((score, -chunk_index, sentence[:700], source))
    candidates.sort(reverse=True)
    parts: list[str] = []
    seen: set[str] = set()
    for _, _, sentence, source in candidates:
        if sentence.casefold() in seen:
            continue
        seen.add(sentence.casefold())
        parts.append(f"{sentence} [{source}]")
        if len(parts) == 3:
            break
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
        if not _has_sufficient_evidence(query, chunks):
            return {
                "answer": NO_EVIDENCE_ANSWER,
                "sources": [],
                "retrieval_source": "none",
                "generation_mode": "local_safe_rejection",
            }
        return {
            "answer": _extractive_answer(query, chunks),
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
