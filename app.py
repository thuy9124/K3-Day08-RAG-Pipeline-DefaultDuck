"""
🇻🇳 GenZ LaborLaw AI — Trợ Lý Tra Cứu Pháp Luật Lao Động Việt Nam
Streamlit App tích hợp giao diện HTML/CSS/JS tùy biến (trolyluat.vn Style - Hướng 2).

Chạy:
    python -m streamlit run app.py
"""

import os
import sys
import socket
import threading
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()

# Thêm project root vào sys.path để import các task từ src/
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# TỰ ĐỘNG KHỞI CHẠY FASTAPI BACKGROUND SERVER (NẾU CHƯA MỞ PORT 8000)
# =============================================================================

def ensure_api_running():
    """Tự động mở Uvicorn API server nếu port 8000 chưa hoạt động."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', 8000))
    sock.close()
    if result != 0:
        try:
            import uvicorn
            from api import app as fastapi_app
            thread = threading.Thread(
                target=uvicorn.run,
                kwargs={"app": fastapi_app, "host": "127.0.0.1", "port": 8000, "log_level": "error"},
                daemon=True
            )
            thread.start()
        except Exception as e:
            print(f"Lỗi khởi chạy FastAPI background: {e}")

ensure_api_running()

# =============================================================================
# STREAMLIT PAGE CONFIG & FULLSCREEN CSS RESET
# =============================================================================

st.set_page_config(
    page_title="Trợ Lý Pháp Luật Lao Động Việt Nam AI (trolyluat.vn)",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide default Streamlit padding, header, footer to allow 100% viewport iframe
st.markdown("""
<style>
    header[data-testid="stHeader"] { display: none !important; }
    .stApp > header { display: none !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        max-width: 100% !important;
    }
    iframe {
        width: 100% !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD HTML / CSS / JS FRONTEND CODEBASE
# =============================================================================

def load_custom_frontend() -> str:
    """Tải và nhúng trực tiếp HTML, CSS, JS từ thư mục frontend/."""
    frontend_dir = PROJECT_ROOT / "frontend"
    index_path = frontend_dir / "index.html"
    css_path = frontend_dir / "style.css"
    js_path = frontend_dir / "app.js"

    if not index_path.exists() or not css_path.exists() or not js_path.exists():
        return "<h3>❌ Thư mục frontend/ thiếu file index.html, style.css hoặc app.js.</h3>"

    html_content = index_path.read_text(encoding="utf-8")
    css_content = css_path.read_text(encoding="utf-8")
    js_content = js_path.read_text(encoding="utf-8")

    # Nhúng trực tiếp CSS vào HTML style tag
    html_content = html_content.replace(
        '<link rel="stylesheet" href="/static/style.css">',
        f'<style>\n{css_content}\n</style>'
    )

    # Đảm bảo Javascript gọi endpoint API http://127.0.0.1:8000/api/chat
    js_content_modified = js_content.replace('fetch("/api/chat"', 'fetch("http://127.0.0.1:8000/api/chat"')

    # Nhúng trực tiếp JS vào HTML script tag
    html_content = html_content.replace(
        '<script src="/static/app.js"></script>',
        f'<script>\n{js_content_modified}\n</script>'
    )

    return html_content

# =============================================================================
# RENDER CUSTOM FRONTEND COMPONENT
# =============================================================================

custom_html = load_custom_frontend()
components.html(custom_html, height=920, scrolling=True)





