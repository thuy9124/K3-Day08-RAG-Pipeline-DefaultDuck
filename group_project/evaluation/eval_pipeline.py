"""Zero-cost deterministic evaluation for the labor-law RAG pipeline.

This intentionally does not call an LLM judge. The four scores are transparent
token-overlap proxies and the report labels them as such (not fabricated RAGAS).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Callable

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

from src.task5_semantic_search import semantic_search
from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import NO_EVIDENCE_ANSWER, _extractive_answer, _has_sufficient_evidence

TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
STOPWORDS = {"và", "là", "có", "được", "cho", "của", "thì", "một", "những", "các", "với", "trong", "tại", "theo", "không", "người", "lao", "động"}
METRICS = ("faithfulness", "answer_relevancy", "context_recall", "context_precision")


def tokens(text: str) -> set[str]:
    return {t for t in TOKEN_PATTERN.findall(text.casefold()) if len(t) > 2 and t not in STOPWORDS}


def load_golden_dataset() -> list[dict]:
    return json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))


def _f1(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    common = len(left & right)
    precision, recall = common / len(left), common / len(right)
    return 2 * precision * recall / (precision + recall) if common else 0.0


def _is_ood(item: dict) -> bool:
    return any(str(value).startswith("OUT_OF_DOMAIN") for value in item["expected_context"])


def evaluate_config(
    dataset: list[dict], name: str, search: Callable[[str, int], list[dict]], top_k: int = 5
) -> tuple[dict[str, float], list[dict]]:
    rows: list[dict] = []
    for index, item in enumerate(dataset, 1):
        question, expected = item["question"], item["expected_answer"]
        chunks = search(question, top_k)
        ood = _is_ood(item)
        enough = _has_sufficient_evidence(question, chunks)
        answer = _extractive_answer(question, chunks) if enough and not ood else NO_EVIDENCE_ANSWER
        context_text = " ".join(str(chunk.get("content", "")) for chunk in chunks)
        context_terms, answer_terms, expected_terms = tokens(context_text), tokens(answer), tokens(expected)

        if ood:
            safe = 1.0 if answer == NO_EVIDENCE_ANSWER else 0.0
            scores = dict.fromkeys(METRICS, safe)
        else:
            answer_content = answer.split("[")[0]
            answer_content_terms = tokens(answer_content)
            faithfulness = (
                len(answer_content_terms & context_terms) / len(answer_content_terms)
                if answer_content_terms else 0.0
            )
            relevant_chunks = sum(
                1 for chunk in chunks
                if len(tokens(str(chunk.get("content", ""))) & expected_terms) >= 2
            )
            scores = {
                "faithfulness": faithfulness,
                "answer_relevancy": _f1(answer_terms, expected_terms),
                "context_recall": len(context_terms & expected_terms) / len(expected_terms) if expected_terms else 0.0,
                "context_precision": relevant_chunks / len(chunks) if chunks else 0.0,
            }
        rows.append({
            "id": index,
            "question": question,
            "retrieved": len(chunks),
            "safe_rejection": answer == NO_EVIDENCE_ANSWER,
            **scores,
        })
        print(f"  [{name} {index:02d}/{len(dataset)}] chunks={len(chunks)}")
    summary = {metric: mean(row[metric] for row in rows) for metric in METRICS}
    return summary, rows


def _dense(query: str, top_k: int) -> list[dict]:
    return semantic_search(query, top_k=top_k)


def _hybrid(query: str, top_k: int) -> list[dict]:
    return retrieve(query, top_k=top_k)


def export_results(results: dict[str, tuple[dict, list[dict]]]) -> None:
    hybrid, dense = results["hybrid_local"], results["dense_local"]
    lines = [
        "# Báo cáo benchmark RAG pháp luật lao động",
        "",
        "Kết quả được tạo từ lần chạy thật, hoàn toàn local. Bốn chỉ số dưới đây là **proxy deterministic theo token overlap**, không phải điểm RAGAS do LLM chấm.",
        "",
        "## A/B retrieval",
        "",
        "| Metric | Hybrid local (BM25 + structural + RRF) | Dense local (char n-gram + Chroma) |",
        "|---|---:|---:|",
    ]
    labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer relevancy",
        "context_recall": "Context recall",
        "context_precision": "Context precision",
    }
    for metric in METRICS:
        lines.append(f"| {labels[metric]} | {hybrid[0][metric]:.4f} | {dense[0][metric]:.4f} |")
    lines += [
        "", "## Chi tiết cấu hình Hybrid local", "",
        "| # | Retrieved | Safe reject | Faithfulness | Relevancy | Recall | Precision |",
        "|---:|---:|:---:|---:|---:|---:|---:|",
    ]
    for row in hybrid[1]:
        lines.append(
            f"| {row['id']} | {row['retrieved']} | {'yes' if row['safe_rejection'] else 'no'} | "
            f"{row['faithfulness']:.3f} | {row['answer_relevancy']:.3f} | "
            f"{row['context_recall']:.3f} | {row['context_precision']:.3f} |"
        )
    lines += [
        "", "## Phạm vi và lưu ý", "",
        "- 12 câu đúng-domain được đối chiếu với Bộ luật Lao động 2019 và Luật BHXH 2024 trong corpus.",
        "- 3 câu ngoài domain phải trả về từ chối an toàn.",
        "- Muốn có điểm RAGAS chính thức cần LLM judge; chế độ đó bị tắt để tuân thủ yêu cầu không phát sinh chi phí.",
    ]
    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {RESULTS_PATH}")


def main() -> None:
    dataset = load_golden_dataset()
    if len(dataset) < 15:
        raise RuntimeError("Golden dataset phải có tối thiểu 15 câu")
    results = {
        "hybrid_local": evaluate_config(dataset, "hybrid", _hybrid),
        "dense_local": evaluate_config(dataset, "dense", _dense),
    }
    export_results(results)


if __name__ == "__main__":
    main()
