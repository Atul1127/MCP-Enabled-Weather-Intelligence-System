"""Stage-by-stage benchmark for the local weather retrieval stack."""
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


def _timed(fn):
    started = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - started) * 1000


def benchmark_cases(cases: list[dict[str, Any]], store: Any) -> dict[str, Any]:
    stage_rows: dict[str, list[dict[str, Any]]] = {stage: [] for stage in STAGES}
    for case in cases:
        query = case["question"]
        gold = str(case["gold_source"])
        allowed = store.filtered_rows()
        dense, dense_ms = _timed(lambda: dense_search(store, query, CANDIDATE_K, allowed))
        sparse, sparse_ms = _timed(lambda: sparse_search(store, query, CANDIDATE_K, allowed))
        hybrid, hybrid_ms = _timed(lambda: fuse(dense, sparse, CANDIDATE_K))
        reranked, rerank_ms = _timed(lambda: rerank(query, hybrid, CANDIDATE_K))
        mmr, mmr_ms = _timed(lambda: select_mmr(reranked, FINAL_K, lambda_mult=0.75))
        cumulative = {
            "dense": dense_ms,
            "bm25": dense_ms + sparse_ms,
            "hybrid": dense_ms + sparse_ms + hybrid_ms,
            "reranked": dense_ms + sparse_ms + hybrid_ms + rerank_ms,
            "reranked_mmr": dense_ms + sparse_ms + hybrid_ms + rerank_ms + mmr_ms,
        }
        results = {
            "dense": (dense[:FINAL_K], dense_ms),
            "bm25": (sparse[:FINAL_K], sparse_ms),
            "hybrid": (hybrid[:FINAL_K], hybrid_ms),
            "reranked": (reranked[:FINAL_K], rerank_ms),
            "reranked_mmr": (mmr, mmr_ms),
        }
        for stage, (documents, isolated_ms) in results.items():
            stage_rows[stage].append({
                "id": case["id"],
                "category": case["category"],
                "gold_source": gold,
                "latency_ms": round(isolated_ms, 3),
                "cumulative_latency_ms": round(cumulative[stage], 3),
                **_metrics(documents, gold),
            })

    summary: dict[str, dict[str, float | int]] = {}
    for stage, rows in stage_rows.items():
        isolated = [row["latency_ms"] for row in rows]
        cumulative = [row["cumulative_latency_ms"] for row in rows]
        summary[stage] = {
            "cases": len(rows),
            "hit_at_1": round(statistics.mean(r["hit_at_1"] for r in rows), 4),
            "hit_at_5": round(statistics.mean(r["hit_at_5"] for r in rows), 4),
            "mrr": round(statistics.mean(r["mrr"] for r in rows), 4),
            "ndcg_at_5": round(statistics.mean(r["ndcg_at_5"] for r in rows), 4),
            "mean_latency_ms": round(statistics.mean(isolated), 3),
            "p50_latency_ms": round(statistics.median(isolated), 3),
            "p95_latency_ms": round(percentile(isolated, 0.95), 3),
            "mean_cumulative_latency_ms": round(statistics.mean(cumulative), 3),
            "p50_cumulative_latency_ms": round(statistics.median(cumulative), 3),
            "p95_cumulative_latency_ms": round(percentile(cumulative, 0.95), 3),
        }
    return {
        "dataset": DATASET.name,
        "candidate_k": CANDIDATE_K,
        "final_k": FINAL_K,
        "metric_definition": "One canonical gold document per case; hit@K is document recall@K.",
        "latency_definition": "latency_ms is isolated stage cost; cumulative_latency_ms is end-to-end cost through that stage.",
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
