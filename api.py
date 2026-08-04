"""
🇻🇳 FastAPI Backend Server — GenZ LaborLaw RAG AI
Exposes REST API endpoints for the HTML/CSS/JS Frontend.

Chạy:
    .venv\\Scripts\\python -m uvicorn api:app --reload --port 8000
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import task10 generation function
from src.task10_generation import generate_with_citation

app = FastAPI(
    title="GenZ LaborLaw AI API",
    description="REST API Server cho hệ thống RAG Tra Cứu Pháp Luật Lao Động Việt Nam",
    version="2.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi tra cứu pháp luật lao động")
    top_k: int = Field(default=5, ge=1, le=10, description="Số lượng đoạn trích dẫn")


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "GenZ LaborLaw AI API Server is running"}


@app.post("/api/chat")
def chat_endpoint(request: QueryRequest):
    """
    RAG Chat Endpoint kết nối với Task 10 generation.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        response = generate_with_citation(query=request.query, top_k=request.top_k)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi thực thi RAG Pipeline: {str(e)}")


# Serve static frontend files if folder exists
FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def read_index():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "Frontend index.html missing"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
