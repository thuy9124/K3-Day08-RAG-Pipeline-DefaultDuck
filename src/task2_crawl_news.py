"""Crawler bài viết công khai về pháp luật lao động Việt Nam.

Tuân thủ robots.txt, không dùng trình duyệt, lưu từng bài JSON cho pipeline hiện
tại và đồng thời xuất bộ dữ liệu tổng hợp JSONL/CSV.
"""
from __future__ import annotations

import csv
import argparse
import json
import logging
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

MAX_ARTICLES = 15
MIN_REQUIRED_ARTICLES = 5
REQUEST_DELAY_SECONDS = 1.5
TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
MIN_CONTENT_LENGTH = 300

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
NEWS_DIR = DATA_DIR / "landing" / "news"
LOG_DIR = PROJECT_DIR / "logs"
JSONL_PATH = DATA_DIR / "labor_law_articles.jsonl"
CSV_PATH = DATA_DIR / "labor_law_articles.csv"

USER_AGENT = "LaborLawResearchBot/1.0 (educational research crawler)"
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "vi-VN,vi;q=0.9"}

ARTICLE_URLS = [
    "https://baochinhphu.vn/thoi-gian-thu-viec-khong-phai-tham-gia-bhxh-102240122113742016.htm",
    "https://xaydungchinhsach.chinhphu.vn/co-phai-bat-buoc-thuong-tet-tien-luong-lam-them-gio-ngay-le-tet-duoc-tinh-the-nao-119240201145549.htm",
    "https://baochinhphu.vn/hieu-the-nao-ve-quy-dinh-tinh-gop-ngay-phep-nam-102240613101442693.htm",
    "https://xaydungchinhsach.chinhphu.vn/nguoi-lao-dong-nghi-viec-duoc-nhan-nhung-khoan-tien-nao-119240509063307003.htm",
    "https://xaydungchinhsach.chinhphu.vn/noi-dung-cua-hop-dong-lao-dong-va-hop-dong-lam-viec-co-gi-khac-nhau-119230830114203459.htm",
    "https://xaydungchinhsach.chinhphu.vn/lao-dong-nu-duoc-huong-nhung-quyen-loi-gi-ve-lao-dong-va-bhxh-119240309012658792.htm",
    "https://xaydungchinhsach.chinhphu.vn/nghi-dinh-so-293-2025-nd-cp-quy-dinh-muc-luong-toi-thieu-doi-voi-nguoi-lao-dong-lam-viec-theo-hop-dong-lao-dong-119251110172808433.htm",
    "https://baochinhphu.vn/lam-them-gio-the-nao-la-dung-quy-dinh-102240130085419703.htm",
    "https://baochinhphu.vn/khi-nao-doanh-nghiep-duoc-dang-ky-tang-gio-lam-them-102230316160746128.htm",
    "https://baochinhphu.vn/di-nghia-vu-quan-su-co-duoc-tinh-vao-thoi-gian-cong-tac-102260309110857267.htm",
    "https://baochinhphu.vn/lao-dong-hop-dong-nghi-viec-khong-luong-co-duoc-ho-tro-khong-102304848.htm",
    "https://baochinhphu.vn/dong-bhxh-cho-thoi-gian-nghi-viec-truoc-day-the-nao-102265214.htm",
    "https://baochinhphu.vn/thu-tuc-doi-voi-lao-dong-nuoc-ngoai-lam-viec-ngan-han-102260401093539635.htm",
    "https://baochinhphu.vn/trinh-tu-thu-tuc-cap-giay-phep-lao-dong-cho-nguoi-nuoc-ngoai-102260107101633495.htm",
    "https://xaydungchinhsach.chinhphu.vn/quy-dinh-hop-dong-lao-dong-doi-voi-giang-vien-dong-co-huu-119260521095240774.htm",
    "https://xaydungchinhsach.chinhphu.vn/quy-dinh-hop-dong-lam-viec-doi-voi-vien-chuc-cham-dut-hop-dong-119251231110414711.htm",
    "https://xaydungchinhsach.chinhphu.vn/de-xuat-dieu-kien-hinh-thuc-hop-dong-voi-nha-giao-hop-dong-toan-thoi-gian-sau-khi-nghi-huu-11926032415244989.htm",
    "https://xaydungchinhsach.chinhphu.vn/hop-dong-cong-viec-trong-don-vi-su-nghiep-cong-lap-hinh-thuc-loai-va-thoi-han-hop-dong-119260630162000019.htm",
    "https://xaydungchinhsach.chinhphu.vn/quy-dinh-moi-ve-tien-luong-co-hieu-luc-tu-1-7-2026-119260626112758079.htm",
    "https://xaydungchinhsach.chinhphu.vn/toan-van-luat-cong-doan-119241217155547243.htm",
]

CONTENT_SELECTORS = (
    '[itemprop="articleBody"]', ".article-body", ".article__body",
    ".detail-content", ".detail__content", ".news-content",
    ".content-detail", "article", "main",
)
DROP_TAGS = ("script", "style", "noscript", "svg", "form", "nav", "footer", "aside", "iframe")
NOISE_PATTERN = re.compile(
    r"breadcrumb|related|recommend|share|social|advert|banner|menu|navigation|"
    r"tag-list|box-tin|tin-lien-quan|tin-khac|doc-them|xem-them|copyright",
    re.IGNORECASE,
)
NAVIGATION_TEXT = re.compile(
    r"^(?:(đọc thêm|xem thêm|tin liên quan|bài liên quan|chia sẻ|theo dõi|quảng cáo)\b|"
    r"(?:www\.)?chinhphu\.vn\s*$)",
    re.IGNORECASE,
)
LABOR_KEYWORDS = (
    "lao động", "hợp đồng", "thử việc", "tiền lương", "làm thêm giờ",
    "nghỉ phép", "nghỉ việc", "sa thải", "người lao động",
)

logger = logging.getLogger("labor_law_crawler")
_robots_cache: dict[str, RobotFileParser | None] = {}


class CrawlError(RuntimeError):
    """Lỗi có thể bỏ qua đối với một URL riêng lẻ."""


def setup_logging() -> None:
    """Khởi tạo thư mục output và logging ra console/file."""
    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(LOG_DIR / "crawl.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def create_session() -> requests.Session:
    """Tạo HTTP session có retry giới hạn cho 429 và lỗi 5xx."""
    retry = Retry(
        total=MAX_RETRIES, connect=MAX_RETRIES, read=MAX_RETRIES,
        status=MAX_RETRIES, backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}), respect_retry_after_header=True,
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update(HEADERS)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def load_seed_urls() -> list[str]:
    """Trả danh sách seed, giữ nguyên thứ tự và loại URL trùng."""
    return list(dict.fromkeys(ARTICLE_URLS))


def check_robots_allowed(url: str, session: requests.Session) -> bool:
    """Kiểm tra và cache robots.txt theo origin."""
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if origin not in _robots_cache:
        robots_url = f"{origin}/robots.txt"
        logger.info("Đang kiểm tra robots.txt: %s", robots_url)
        try:
            response = session.get(robots_url, timeout=TIMEOUT_SECONDS)
            if response.status_code == 404:
                _robots_cache[origin] = None
            elif response.status_code in (401, 403, 429):
                raise CrawlError(f"robots.txt trả HTTP {response.status_code}")
            else:
                response.raise_for_status()
                parser = RobotFileParser()
                parser.set_url(robots_url)
                parser.parse(response.text.splitlines())
                _robots_cache[origin] = parser
        except CrawlError:
            raise
        except requests.RequestException as exc:
            raise CrawlError(f"không tải được robots.txt: {exc}") from exc
    parser = _robots_cache[origin]
    allowed = parser is None or parser.can_fetch(USER_AGENT, url)
    if not allowed:
        logger.warning("URL bị robots.txt từ chối: %s", url)
    return allowed


def fetch_html(url: str, session: requests.Session) -> tuple[str, str]:
    """Tải HTML và trả về HTML cùng URL cuối sau redirect."""
    logger.info("Đang crawl URL: %s", url)
    try:
        response = session.get(url, timeout=TIMEOUT_SECONDS, allow_redirects=True)
        logger.info("HTTP status: %s", response.status_code)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise CrawlError(f"lỗi HTTP/kết nối: {exc}") from exc
    if "html" not in response.headers.get("Content-Type", "").lower():
        raise CrawlError("response không phải HTML")
    response.encoding = response.apparent_encoding or response.encoding
    return response.text, response.url


def normalize_inline(text: str) -> str:
    """Chuẩn hóa Unicode và khoảng trắng trên một dòng."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def _meta_content(soup: BeautifulSoup, *names: str) -> str:
    for name in names:
        tag = soup.find("meta", attrs={"property": name})
        tag = tag or soup.find("meta", attrs={"name": name})
        tag = tag or soup.find("meta", attrs={"itemprop": name})
        if tag and tag.get("content"):
            return normalize_inline(str(tag["content"]))
    return ""


def extract_title(soup: BeautifulSoup) -> str:
    """Trích title theo meta OG, Twitter, h1 rồi title."""
    title = _meta_content(soup, "og:title", "twitter:title")
    if not title:
        tag = soup.find("h1") or soup.find("title")
        title = normalize_inline(tag.get_text(" ", strip=True)) if tag else ""
    return re.sub(
        r"\s*[-|–—]\s*(Báo Chính Phủ|Cổng Thông tin điện tử Chính phủ).*$",
        "", title, flags=re.IGNORECASE,
    ).strip()


def extract_published_at(soup: BeautifulSoup) -> str:
    """Trích ngày đăng nếu trang cung cấp, không tự suy đoán."""
    value = _meta_content(
        soup, "article:published_time", "datePublished", "pubdate", "publishdate"
    )
    if value:
        return value
    time_tag = soup.find("time")
    if time_tag:
        return normalize_inline(str(time_tag.get("datetime") or time_tag.get_text(" ", strip=True)))
    head_text = normalize_inline(soup.get_text(" ", strip=True)[:2500])
    match = re.search(r"\b\d{1,2}/\d{1,2}/\d{4}(?:\s+\d{1,2}:\d{2})?\b", head_text)
    return match.group(0) if match else ""


def _block_score(block: Tag) -> float:
    paragraphs = block.find_all("p")
    paragraph_text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
    if len(paragraphs) < 2 or len(paragraph_text) < MIN_CONTENT_LENGTH:
        return -1
    link_text = " ".join(a.get_text(" ", strip=True) for a in block.find_all("a"))
    return len(paragraph_text) - 0.5 * len(link_text)


def _choose_content_block(soup: BeautifulSoup) -> Tag:
    for selector in CONTENT_SELECTORS:
        for candidate in soup.select(selector):
            if _block_score(candidate) >= MIN_CONTENT_LENGTH:
                return candidate
    candidates = soup.find_all(("article", "main", "section", "div"))
    if not candidates:
        raise CrawlError("HTML không có khối nội dung phù hợp")
    best = max(candidates, key=_block_score)
    if _block_score(best) < MIN_CONTENT_LENGTH:
        raise CrawlError("không tìm thấy nội dung chính đủ dài")
    return best


def clean_content(block: Tag) -> str:
    """Loại thành phần điều hướng, chuẩn hóa và khử trùng đoạn."""
    for tag in block.find_all(DROP_TAGS):
        tag.decompose()
    for tag in list(block.find_all(True)):
        # Một node cha bị decompose() sẽ làm rỗng attrs của các node con đã có
        # trong snapshot; bỏ qua các node không còn thuộc cây DOM.
        if tag.parent is None or tag.attrs is None:
            continue
        marker = " ".join(tag.get("class", [])) + " " + str(tag.get("id", ""))
        if NOISE_PATTERN.search(marker):
            tag.decompose()
    paragraphs: list[str] = []
    seen: set[str] = set()
    for element in block.find_all(("h2", "h3", "h4", "p", "li", "blockquote")):
        text = normalize_inline(element.get_text(" ", strip=True))
        if not text or NAVIGATION_TEXT.search(text):
            continue
        link_text = normalize_inline(" ".join(a.get_text(" ", strip=True) for a in element.find_all("a")))
        if link_text and link_text == text:
            continue
        key = text.casefold()
        if key not in seen:
            seen.add(key)
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def extract_main_content(soup: BeautifulSoup) -> str:
    """Trích xuất phần thân bài chính đã làm sạch."""
    for tag in soup.find_all(DROP_TAGS):
        tag.decompose()
    return clean_content(_choose_content_block(soup))


def detect_topic(title: str, content: str) -> str:
    """Gán chủ đề bằng quy tắc từ khóa."""
    title_text = title.casefold()
    full_text = f"{title}\n{content}".casefold()
    rules = (
        (("thử việc",), "thu_viec"),
        (("làm thêm giờ", "tăng ca"), "lam_them_gio"),
        (("nghỉ phép", "phép năm"), "nghi_phep"),
        (("sa thải", "kỷ luật lao động"), "sa_thai"),
        (("chấm dứt hợp đồng", "nghỉ việc"), "cham_dut_hop_dong"),
        (("lương tối thiểu", "mức lương tối thiểu"), "luong_toi_thieu"),
        (("hợp đồng lao động",), "hop_dong_lao_dong"),
    )
    # Từ khóa trong tiêu đề thể hiện chủ đề chính tốt hơn một đề cập phụ ở thân bài.
    for text in (title_text, full_text):
        for keywords, topic in rules:
            if any(keyword in text for keyword in keywords):
                return topic
    return "phap_luat_lao_dong"


def validate_article(article: dict[str, Any]) -> None:
    """Xác thực record theo các tiêu chí trong đặc tả."""
    if not article.get("url") or not article.get("title"):
        raise CrawlError("thiếu URL cuối hoặc tiêu đề")
    content = str(article.get("content", ""))
    if len(content) < MIN_CONTENT_LENGTH:
        raise CrawlError(f"nội dung quá ngắn ({len(content)} ký tự)")
    if len([part for part in content.split("\n\n") if part.strip()]) < 2:
        raise CrawlError("nội dung có ít hơn 2 đoạn")
    combined = f"{article['title']} {content}".casefold()
    if not any(keyword in combined for keyword in LABOR_KEYWORDS):
        raise CrawlError("không tìm thấy từ khóa pháp luật lao động")


def crawl_article(url: str, session: requests.Session, article_id: int) -> dict[str, Any]:
    """Crawl, parse và xác thực một bài viết."""
    if not check_robots_allowed(url, session):
        raise CrawlError("robots.txt không cho phép crawl")
    html, final_url = fetch_html(url, session)
    soup = BeautifulSoup(html, "html.parser")
    title = extract_title(soup)
    published_at = extract_published_at(soup)
    content = extract_main_content(soup)
    crawled_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    article: dict[str, Any] = {
        "id": f"labor-law-{article_id:04d}",
        "source": urlparse(final_url).netloc,
        "title": title,
        "published_at": published_at,
        "url": final_url,
        "topic": detect_topic(title, content),
        "content": content,
        "content_length": len(content),
        "crawled_at": crawled_at,
        # Alias tương thích với task 3/template cũ của project.
        "date_crawled": crawled_at,
        "content_markdown": content,
    }
    validate_article(article)
    logger.info("Tiêu đề: %s", title)
    logger.info("Độ dài nội dung: %s ký tự", len(content))
    return article


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Ghi JSON Lines UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_csv(records: list[dict[str, Any]], path: Path) -> None:
    """Ghi CSV UTF-8-SIG để Excel đọc đúng tiếng Việt."""
    fields = [
        "id", "source", "title", "published_at", "url", "topic",
        "content", "content_length", "crawled_at",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def save_individual_json(records: list[dict[str, Any]]) -> None:
    """Ghi mỗi bài một JSON vào landing/news."""
    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records, 1):
        path = NEWS_DIR / f"labor_law_{index:02d}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def crawl_all() -> list[dict[str, Any]]:
    """Crawl lần lượt đến khi đủ MAX_ARTICLES hoặc hết seed URL."""
    setup_logging()
    session = create_session()
    records: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    urls = load_seed_urls()
    try:
        for index, url in enumerate(urls):
            if len(records) >= MAX_ARTICLES:
                break
            try:
                article = crawl_article(url, session, len(records) + 1)
                normalized_url = article["url"].rstrip("/")
                normalized_title = article["title"].casefold()
                if normalized_url in seen_urls or normalized_title in seen_titles:
                    raise CrawlError("bài viết trùng URL hoặc tiêu đề")
                records.append(article)
                seen_urls.add(normalized_url)
                seen_titles.add(normalized_title)
                logger.info("Crawl thành công %s/%s bài", len(records), MAX_ARTICLES)
            except (CrawlError, ValueError, AttributeError) as exc:
                logger.error("Bỏ qua URL %s — %s", url, exc)
            if index < len(urls) - 1 and len(records) < MAX_ARTICLES:
                time.sleep(REQUEST_DELAY_SECONDS)
    finally:
        session.close()
    save_individual_json(records)
    save_jsonl(records, JSONL_PATH)
    save_csv(records, CSV_PATH)
    logger.info("Đã ghi JSONL, CSV và %s file JSON", len(records))
    return records


def load_cached_articles() -> list[dict[str, Any]]:
    """Đọc và kiểm tra corpus đã crawl, không thực hiện request mạng."""
    records: list[dict[str, Any]] = []
    if not NEWS_DIR.exists():
        return records
    for path in sorted(NEWS_DIR.glob("*.json")):
        try:
            article = json.loads(path.read_text(encoding="utf-8"))
            validate_article(article)
            records.append(article)
        except (OSError, json.JSONDecodeError, CrawlError) as exc:
            logger.warning("Bỏ qua cache không hợp lệ %s — %s", path.name, exc)
    return records


def main() -> int:
    """Mặc định chỉ validate cache; thêm ``--refresh`` khi thật sự muốn crawl mạng."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="crawl lại tối đa 15 bài")
    args = parser.parse_args()
    setup_logging()
    records = crawl_all() if args.refresh else load_cached_articles()
    if len(records) < MIN_REQUIRED_ARTICLES:
        logger.error(
            "Chỉ crawl được %s bài; yêu cầu tối thiểu %s bài hợp lệ",
            len(records), MIN_REQUIRED_ARTICLES,
        )
        return 1
    mode = "crawl mới" if args.refresh else "cache hiện có"
    logger.info("Hoàn tất kiểm tra %s: %s bài hợp lệ", mode, len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
