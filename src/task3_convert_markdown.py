"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install "markitdown[pdf]"
    # Lưu ý: cần extra [pdf] để convert được file PDF. Chỉ "pip install markitdown"
    # (không có extra) sẽ báo MissingDependencyException khi convert PDF, dù JSON/DOCX
    # vẫn convert bình thường.

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import argparse
from pathlib import Path

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from markitdown import MarkItDown
    except ImportError as exc:
        raise RuntimeError('Cần cài "markitdown[pdf]" khi dùng --refresh') from exc
    md = MarkItDown()

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            # TODO: Convert và lưu file
            result = md.convert(str(filepath))
            output_path = output_dir / f"{filepath.stem}.md"
            converted = (result.text_content or "").strip()
            if not converted:
                if output_path.exists() and output_path.stat().st_size > 200:
                    print(f"  ! Converter trả rỗng; giữ bản Markdown hiện có: {output_path}")
                    continue
                raise RuntimeError(f"Không trích xuất được nội dung từ {filepath.name}")
            output_path.write_text(converted, encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            # TODO: Đọc JSON, extract content_markdown, lưu thành .md
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"

            # Thêm metadata header
            header = f"# {data.get('title', 'Unknown')}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

            content = header + data.get("content_markdown", "")
            output_path.write_text(content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


def validate_standardized() -> list[Path]:
    """Xác nhận mỗi source có một Markdown không rỗng mà không chạy converter."""
    missing: list[str] = []
    outputs: list[Path] = []
    for source in sorted(LANDING_DIR.rglob("*")):
        if not source.is_file() or source.suffix.lower() not in {".pdf", ".doc", ".docx", ".json"}:
            continue
        relative = source.relative_to(LANDING_DIR).with_suffix(".md")
        output = OUTPUT_DIR / relative
        if not output.exists() or output.stat().st_size <= 200:
            missing.append(relative.as_posix())
        else:
            outputs.append(output)
    if missing:
        raise RuntimeError("Thiếu Markdown hợp lệ: " + ", ".join(missing))
    print(f"✓ Đã kiểm tra {len(outputs)} file Markdown tương ứng, tất cả đều hợp lệ")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="convert lại source files")
    args = parser.parse_args()
    convert_all() if args.refresh else validate_standardized()
