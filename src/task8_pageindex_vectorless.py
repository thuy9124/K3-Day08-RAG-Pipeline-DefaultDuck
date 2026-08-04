"""Task 8 — vectorless structural fallback với interface PageIndex.

Nếu PageIndex cloud/SDK chưa được cấu hình, module xây cây section từ Markdown và
truy hồi local theo heading + nội dung. Fallback này không dùng embedding/Chroma.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

STANDARDIZED_DIR = Path(__file__).resolve().parent.parent / "data" / "standardized"
LOCAL_INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "pageindex" / "structural_index.json"
TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(text.casefold()))


def _parse_markdown(path: Path) -> list[dict[str, Any]]:
    """Tách Markdown thành section theo heading, giữ đoạn văn nguyên vẹn."""
    text = path.read_text(encoding="utf-8-sig", errors="replace").strip()
    if not text:
        return []
    sections: list[dict[str, Any]] = []
    heading = path.stem
    buffer: list[str] = []

    def flush() -> None:
        content = "\n".join(buffer).strip()
        if content:
            sections.append({
                "title": heading,
                "content": content,
                "source": path.name,
                "source_path": path.relative_to(STANDARDIZED_DIR).as_posix(),
            })
        buffer.clear()

    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match:
            flush()
            heading = match.group(1).strip()
        else:
            buffer.append(line)
    flush()
    return sections


@lru_cache(maxsize=1)
def build_structural_index() -> list[dict[str, Any]]:
    """Build section index in memory từ toàn bộ Markdown không rỗng."""
    sections: list[dict[str, Any]] = []
    if STANDARDIZED_DIR.exists():
        for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
            sections.extend(_parse_markdown(path))
    return sections


def upload_documents() -> dict[str, Any]:
    """Persist local structural index; không giả lập upload cloud khi thiếu SDK/key."""
    sections = build_structural_index()
    LOCAL_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_INDEX_PATH.write_text(
        json.dumps(sections, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"mode": "local_structural", "sections": len(sections), "path": str(LOCAL_INDEX_PATH)}


def pageindex_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Vectorless search, ưu tiên token trùng trong heading hơn body."""
    query = query.strip()
    query_tokens = _tokens(query)
    if not query_tokens or top_k <= 0:
        return []
    scored: list[tuple[float, dict[str, Any]]] = []
    for section in build_structural_index():
        title_tokens = _tokens(section["title"])
        body_tokens = _tokens(section["content"])
        title_hits = len(query_tokens & title_tokens)
        body_hits = len(query_tokens & body_tokens)
        if not title_hits and not body_hits:
            continue
        coverage = len(query_tokens & (title_tokens | body_tokens)) / len(query_tokens)
        phrase_bonus = 1.0 if query.casefold() in section["content"].casefold() else 0.0
        score = 3.0 * title_hits + body_hits + coverage + phrase_bonus
        scored.append((score, section))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["source_path"], pair[1]["title"]))
    return [
        {
            "content": section["content"],
            "score": float(score),
            "metadata": {
                "source": section["source"],
                "source_path": section["source_path"],
                "section": section["title"],
                "mode": "local_structural",
            },
            "source": "pageindex",
        }
        for score, section in scored[:top_k]
    ]


if __name__ == "__main__":
    print(upload_documents())
    for result in pageindex_search("lương thử việc", top_k=3):
        print(f"[{result['score']:.3f}] {result['metadata']['section']}")
