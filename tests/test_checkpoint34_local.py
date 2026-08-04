"""Test Checkpoint 3–4 hoàn toàn local, không gọi dịch vụ bên ngoài."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# Phải đặt trước khi import Task 9/10 vì flag được đọc ở import time.
os.environ["RAG_ALLOW_EXTERNAL_APIS"] = "false"

from src.task7_reranking import rerank, rerank_rrf
from src.task8_pageindex_vectorless import pageindex_search
import src.task9_retrieval_pipeline as task9
import src.task10_generation as task10

TEST_CASES_PATH = Path(__file__).with_name("checkpoint34_test_cases.json")
TEST_CASES = json.loads(TEST_CASES_PATH.read_text(encoding="utf-8"))


def _candidate(name: str, source: str, chunk_index: int, score: float) -> dict:
    return {
        "content": name,
        "score": score,
        "metadata": {
            "source": source,
            "source_path": source,
            "chunk_index": chunk_index,
        },
    }


class TestRRF:
    def test_rrf_exact_formula_and_deduplication(self):
        first = _candidate("A", "a.md", 0, 0.9)
        second = _candidate("B", "b.md", 0, 0.8)

        results = rerank_rrf([[first, second], [second, first]], top_k=2, k=60)

        expected = 1 / 61 + 1 / 62
        assert len(results) == 2
        assert results[0]["score"] == pytest.approx(expected)
        assert results[1]["score"] == pytest.approx(expected)
        assert {result["content"] for result in results} == {"A", "B"}

    def test_default_rerank_interface_returns_top_k(self):
        candidates = [
            _candidate(f"Document {index}", "source.md", index, 1 - index / 10)
            for index in range(6)
        ]
        results = rerank("test", candidates, top_k=3)
        assert len(results) == 3
        assert all("score" in result for result in results)


@pytest.mark.parametrize("case", TEST_CASES, ids=lambda case: case["id"])
def test_structural_fallback_on_labor_law_cases(case):
    """Dữ liệu thật phải trả section local đúng domain, không dùng vector API."""
    results = pageindex_search(case["query"], top_k=5)

    assert len(results) >= case["min_results"]
    assert all(result["source"] == "pageindex" for result in results)
    assert all(result["metadata"]["mode"] == "local_structural" for result in results)
    combined = " ".join(result["content"].casefold() for result in results)
    assert any(term.casefold() in combined for term in case["expected_terms"])


class TestRetrievalRouting:
    def test_high_dense_score_uses_hybrid_rrf(self, monkeypatch):
        dense = [_candidate("dense relevant", "dense.md", 0, 0.82)]
        sparse = [_candidate("sparse relevant", "sparse.md", 0, 7.5)]
        monkeypatch.setattr(task9, "ALLOW_EXTERNAL_APIS", True)
        monkeypatch.setattr(task9, "semantic_search", lambda query, top_k: dense)
        monkeypatch.setattr(task9, "lexical_search", lambda query, top_k: sparse)
        monkeypatch.setattr(
            task9,
            "pageindex_search",
            lambda query, top_k: pytest.fail("Fallback không được gọi khi cosine cao"),
        )

        results = task9.retrieve("relevant query", top_k=2, score_threshold=0.48)

        assert len(results) == 2
        assert all(result["source"] == "hybrid" for result in results)
        assert all(result["dense_best_score"] == pytest.approx(0.82) for result in results)

    def test_low_dense_score_uses_local_fallback(self, monkeypatch):
        dense = [_candidate("weak dense", "dense.md", 0, 0.30)]
        fallback = [{
            "content": "fallback evidence",
            "score": 3.0,
            "metadata": {"source": "law.md", "mode": "local_structural"},
            "source": "pageindex",
        }]
        monkeypatch.setattr(task9, "ALLOW_EXTERNAL_APIS", True)
        monkeypatch.setattr(task9, "semantic_search", lambda query, top_k: dense)
        monkeypatch.setattr(task9, "lexical_search", lambda query, top_k: [])
        monkeypatch.setattr(task9, "pageindex_search", lambda query, top_k: fallback)

        results = task9.retrieve("weak query", top_k=3, score_threshold=0.48)

        assert results == fallback
        assert results[0]["source"] == "pageindex"

    def test_zero_cost_mode_never_calls_semantic_api(self, monkeypatch):
        monkeypatch.setattr(task9, "ALLOW_EXTERNAL_APIS", False)
        monkeypatch.setattr(
            task9,
            "semantic_search",
            lambda *args, **kwargs: pytest.fail("Không được gọi Voyage trong local mode"),
        )
        monkeypatch.setattr(task9, "lexical_search", lambda query, top_k: [])
        monkeypatch.setattr(task9, "pageindex_search", lambda query, top_k: [])

        assert task9.retrieve("offline query", top_k=3) == []


class TestGeneration:
    def test_reorder_lost_in_the_middle(self):
        chunks = [{"content": str(index), "score": 1 - index / 10} for index in range(5)]
        reordered = task10.reorder_for_llm(chunks)
        assert [chunk["content"] for chunk in reordered] == ["0", "2", "4", "3", "1"]

    def test_context_contains_source_and_section(self):
        context = task10.format_context([{
            "content": "Nội dung Điều 24",
            "score": 0.9,
            "metadata": {"source": "bo-luat-lao-dong.md", "type": "legal", "section": "Điều 24"},
        }])
        assert "bo-luat-lao-dong.md" in context
        assert "Điều 24" in context

    def test_local_generation_has_citation_and_no_llm(self, monkeypatch):
        chunks = [{
            "content": "Tiền lương thử việc do hai bên thỏa thuận.",
            "score": 4.2,
            "metadata": {"source": "bo-luat-lao-dong.md", "chunk_index": 1},
            "source": "pageindex",
        }]
        monkeypatch.setattr(task10, "ALLOW_EXTERNAL_APIS", False)
        monkeypatch.setattr(task10, "retrieve", lambda query, top_k: chunks)

        result = task10.generate_with_citation("Lương thử việc thế nào?", top_k=3)

        assert result["generation_mode"] == "local_extractive"
        assert "[bo-luat-lao-dong.md]" in result["answer"]
        assert result["retrieval_source"] == "pageindex"
        assert len(result["sources"]) == 1

    def test_no_evidence_returns_safe_answer(self, monkeypatch):
        monkeypatch.setattr(task10, "retrieve", lambda query, top_k: [])
        result = task10.generate_with_citation("Không có bằng chứng")
        assert result["answer"] == task10.NO_EVIDENCE_ANSWER
        assert result["sources"] == []
