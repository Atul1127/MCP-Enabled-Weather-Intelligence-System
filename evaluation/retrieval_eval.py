"""Run deterministic document-level retrieval evaluation."""
from __future__ import annotations
import json
import statistics
import time
from pathlib import Path

from rag.pipeline import RAGPipeline
from evaluation.retrieval_metrics import evaluate_ranked_documents, evaluate_relevant_documents

DATASET = Path(__file__).with_name("rag_eval_dataset.json")
REPORT = Path(__file__).with_name("rag_retrieval_report.json")
KS = (1, 3, 5, 10)


def main() -> None:
    cases = json.loads(DATASET.read_text(encoding="utf-8"))["cases"]
    pipeline = RAGPipeline()
    rows: list[dict] = []
    for case in cases:
        started = time.perf_counter()
        result = pipeline.retrieve(case["question"])
        latency = (time.perf_counter() - started) * 1000
        document_metrics = evaluate_relevant_documents(
            result.documents, case.get("relevant_document_ids", []), KS
        )
        topic_metrics = evaluate_ranked_documents(
            result.documents, case.get("expected_source_topics", []), KS
        )
        rows.append({
            "id": case["id"],
            "latency_ms": round(latency, 2),
            **{f"document_{key}": value for key, value in document_metrics.items()},
            **{f"topic_{key}": value for key, value in topic_metrics.items()},
        })

    summary = {"cases": len(rows), "evaluation": "document-level + topic-proxy"}
    keys = [key for key in rows[0] if key not in {"id"}] if rows else []
    for key in keys:
        values = [float(row[key]) for row in rows]
        summary[f"mean_{key}"] = round(statistics.mean(values), 4) if values else 0.0
    latencies = [row["latency_ms"] for row in rows]
    if latencies:
        ordered = sorted(latencies)
        summary["p50_latency_ms"] = round(statistics.median(ordered), 2)
        summary["p90_latency_ms"] = round(ordered[max(0, int(len(ordered) * 0.9) - 1)], 2)
        summary["p95_latency_ms"] = round(ordered[max(0, int(len(ordered) * 0.95) - 1)], 2)

    payload = {
        "dataset": DATASET.name,
        "metric_note": "Document metrics use canonical relevant_document_ids; topic metrics are retained as a secondary proxy.",
        "summary": summary,
        "rows": rows,
    }
    REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
