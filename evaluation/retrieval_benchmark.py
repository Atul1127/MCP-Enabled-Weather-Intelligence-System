"""Stage-by-stage benchmark for the local weather retrieval stack.

The benchmark compares the retrieval stages independently so improvements can be
measured rather than assumed:

    dense -> BM25 -> hybrid RRF -> hybrid + cross-encoder -> hybrid + reranker + MMR

Each case has one canonical gold document ID. Metrics therefore focus on Hit@1,
Hit@5, MRR and nDCG@5, with latency reported per stage. No Gemini calls are made.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.retrieval_metrics import evaluate_relevant_documents
from local_rag_store import get_store
from rag.retrieval.dense import search as dense_search
from rag.retrieval.sparse import search as sparse_search
from rag.retrieval.hybrid import fuse
from rag.reranking.cross_encoder import rerank
from rag.retrieval.diversity import select_mmr

DATASET = Path(__file__).resolve().parent / "weather_retrieval_dataset.json"
REPORT = Path(__file__).resolve().parent / "retrieval_benchmark_report.json"
CANDIDATE_K = 20
FINAL_K = 10
STAGES = ("dense", "bm25", "hybrid", "reranked", "reranked_mmr")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _metrics(documents: list[dict[str, Any]], gold: str) -> dict[str, float]:
    values = evaluate_relevant_documents(documents, [gold], (1, 5, 10))
    return {
        "hit_at_1": values["recall_at_1"],
        "hit_at_5": values["recall_at_5"],
        "mrr": values["mrr"],
        "ndcg_at_5": values["ndcg_at_5"],
    }


def _retrieve_stages(query: str, store: Any, allowed: list[int]) -> dict[str, list[dict[str, Any]]]:
    dense = dense_search(store, query, CANDIDATE_K, allowed)
    sparse = sparse_search(store, query, CANDIDATE_K, allowed)
    hybrid = fuse(dense, sparse, CANDIDATE_K)
    reranked = rerank(query, hybrid, CANDIDATE_K)
    mmr = select_mmr(reranked, FINAL_K, lambda_mult=0.75)
    return {
        "dense": dense[:FINAL_K],
        "bm25": sparse[:FINAL_K],
        "hybrid": hybrid[:FINAL_K],
        "reranked": reranked[:FINAL_K],
        "reranked_mmr": mmr,
    }


def benchmark_cases(cases: list[dict[str, Any]], store: Any) -> dict[str, Any]:
    stage_rows: dict[str, list[dict[str, Any]]] = {stage: [] for stage in STAGES}

    for case in cases:
        query = case["question"]
        gold = str(case["gold_source"])
        allowed = store.filtered_rows()
        stages = _retrieve_stages(query, store, allowed)

        for stage, documents in stages.items():
            started = time.perf_counter()
            # Metric computation is intentionally timed separately from retrieval.
            metrics = _metrics(documents, gold)
            metric_ms = (time.perf_counter() - started) * 1000
            stage_rows[stage].append({
                "id": case["id"],
                "category": case["category"],
                "gold_source": gold,
                "latency_ms": round(metric_ms, 4),
                **metrics,
            })

    summary: dict[str, dict[str, float | int]] = {}
    for stage, rows in stage_rows.items():
        latencies = [row["latency_ms"] for row in rows]
        summary[stage] = {
            "cases": len(rows),
            "hit_at_1": round(statistics.mean(r["hit_at_1"] for r in rows), 4),
            "hit_at_5": round(statistics.mean(r["hit_at_5"] for r in rows), 4),
            "mrr": round(statistics.mean(r["mrr"] for r in rows), 4),
            "ndcg_at_5": round(statistics.mean(r["ndcg_at_5"] for r in rows), 4),
            "metric_p50_ms": round(statistics.median(latencies), 4) if latencies else 0.0,
            "metric_p95_ms": round(percentile(latencies, 0.95), 4),
        }

    return {
        "dataset": DATASET.name,
        "candidate_k": CANDIDATE_K,
        "final_k": FINAL_K,
        "metric_definition": "One canonical gold document per case; hit@K is document recall@K.",
        "summary": summary,
        "rows": stage_rows,
    }


def main() -> None:
    cases = json.loads(DATASET.read_text(encoding="utf-8"))["cases"]
    report = benchmark_cases(cases, get_store())
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
