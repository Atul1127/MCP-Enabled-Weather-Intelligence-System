"""Run deterministic retrieval evaluation against the checked-in RAG dataset."""
from __future__ import annotations
import json
import statistics
import time
from pathlib import Path

from rag.pipeline import RAGPipeline
from evaluation.retrieval_metrics import evaluate_ranked_documents

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
        metrics = evaluate_ranked_documents(result.documents, case.get("expected_source_topics", []), KS)
        rows.append({"id": case["id"], "latency_ms": round(latency, 2), **metrics})

    summary = {"cases": len(rows)}
    for key in rows[0] if rows else []:
        if key in {"id"}:
            continue
        values = [float(row[key]) for row in rows]
        summary[f"mean_{key}"] = round(statistics.mean(values), 4) if values else 0.0
    latencies = [row["latency_ms"] for row in rows]
    summary["p50_latency_ms"] = round(statistics.median(latencies), 2) if latencies else 0.0

    payload = {
        "dataset": DATASET.name,
        "metric_note": "Recall is reported as topic-hit-at-k because the current dataset labels source topics, not canonical relevant document IDs.",
        "summary": summary,
        "rows": rows,
    }
    REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
