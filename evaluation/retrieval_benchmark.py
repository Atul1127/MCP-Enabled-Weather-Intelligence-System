"""Benchmark the non-KG weather retrieval stack.

Run from the repository root:
    python evaluation/retrieval_benchmark.py

Compares Dense, BM25, standard RRF, confidence-aware RRF and cross-encoder
reranking. Reranker cold-start is separated from warm inference and each
query/strategy is repeated to make latency less sensitive to one noisy run.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advanced_rag import CONFIDENCE_WEIGHT, RRF_K, confidence_aware_rrf, rerank, reranker
from local_rag_store import get_store

EVALUATION_DIR = Path(__file__).resolve().parent
DATASET = EVALUATION_DIR / "weather_retrieval_dataset.json"
REPORT = EVALUATION_DIR / "retrieval_benchmark_report.json"
REPEATS = 3


def rank_gold(rows: list[dict], gold: str) -> int | None:
    for rank, row in enumerate(rows, 1):
        if str(row.get("id")) == gold:
            return rank
    return None


def metrics(rows: list[dict], gold: str) -> dict:
    rank = rank_gold(rows, gold)
    return {
        "rank": rank,
        "hit_at_1": int(rank == 1),
        "recall_at_5": int(rank is not None and rank <= 5),
        "mrr": 0.0 if rank is None else 1.0 / rank,
    }


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def ci95(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        mean = values[0] if values else 0.0
        return mean, mean
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(len(values))
    margin = 1.96 * se
    return mean - margin, mean + margin


def evaluate_strategy(store, strategy: str, query: str, gold: str) -> tuple[dict, float, str | None]:
    started = time.perf_counter()
    error = None
    try:
        if strategy == "Dense":
            result = store.dense_search(query, 5)
        elif strategy == "BM25":
            result = store.bm25_search(query, 5)
        else:
            dense = store.dense_search(query, 20)
            bm25 = store.bm25_search(query, 20)
            if strategy == "RRF":
                result = _standard_rrf([dense, bm25], 5)
            elif strategy == "ConfidenceRRF":
                result = confidence_aware_rrf([("dense", dense), ("bm25", bm25)], 5)
            elif strategy == "Reranker":
                fused = confidence_aware_rrf([("dense", dense), ("bm25", bm25)], 20)
                result = rerank(query, fused, 5)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
    except Exception as exc:
        result = []
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter() - started) * 1000
    row = {
        "query": query,
        "gold_source": gold,
        "strategy": strategy,
        "latency_ms": round(latency_ms, 3),
        "error": error,
    }
    row.update(metrics(result, gold))
    return row, latency_ms, error


def main() -> None:
    cases = json.loads(DATASET.read_text(encoding="utf-8"))["cases"]
    store = get_store()
    strategies = ["Dense", "BM25", "RRF", "ConfidenceRRF", "Reranker"]

    # Load the cross-encoder before timing benchmark queries. Cold model load is
    # reported separately so reranker inference is compared fairly with other
    # retrieval strategies.
    cold_start = time.perf_counter()
    reranker()
    reranker_cold_start_ms = round((time.perf_counter() - cold_start) * 1000, 3)

    # One unmeasured warm-up inference avoids measuring the first prediction's
    # tokenizer/model cache initialization.
    reranker_warmup_start = time.perf_counter()
    warmup_candidates = store.dense_search(cases[0]["question"], 5)
    if warmup_candidates:
        reranker().predict([(cases[0]["question"], str(warmup_candidates[0].get("text", "")))])
    reranker_warmup_ms = round((time.perf_counter() - reranker_warmup_start) * 1000, 3)

    raw_rows: list[dict] = []
    for case in cases:
        for strategy in strategies:
            for repeat in range(1, REPEATS + 1):
                row, _, _ = evaluate_strategy(
                    store, strategy, case["question"], case["gold_source"]
                )
                row.update({
                    "id": case["id"],
                    "category": case["category"],
                    "repeat": repeat,
                })
                raw_rows.append(row)

    # Aggregate each query/strategy using median latency and best-of-repeat
    # retrieval outcome consistency. Metrics are computed over query-level
    # medians, not over repeated rows, preventing one query from being weighted
    # three times in the final score.
    query_rows: list[dict] = []
    for case in cases:
        for strategy in strategies:
            subset = [
                r for r in raw_rows
                if r["id"] == case["id"] and r["strategy"] == strategy
            ]
            latency_values = [r["latency_ms"] for r in subset]
            hit_values = [r["hit_at_1"] for r in subset]
            recall_values = [r["recall_at_5"] for r in subset]
            mrr_values = [r["mrr"] for r in subset]
            errors = [r for r in subset if r["error"]]
            query_rows.append({
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "gold_source": case["gold_source"],
                "strategy": strategy,
                "rank": next((r["rank"] for r in subset if r["rank"] is not None), None),
                "hit_at_1": round(statistics.mean(hit_values), 4),
                "recall_at_5": round(statistics.mean(recall_values), 4),
                "mrr": round(statistics.mean(mrr_values), 4),
                "latency_ms": round(statistics.median(latency_values), 3),
                "latency_ms_min": round(min(latency_values), 3),
                "latency_ms_max": round(max(latency_values), 3),
                "errors": len(errors),
            })

    summary = {}
    for strategy in strategies:
        subset = [r for r in query_rows if r["strategy"] == strategy]
        latencies = [r["latency_ms"] for r in subset]
        hit = [r["hit_at_1"] for r in subset]
        mrr = [r["mrr"] for r in subset]
        hit_ci = ci95(hit)
        mrr_ci = ci95(mrr)
        summary[strategy] = {
            "count": len(subset),
            "Hit@1": round(statistics.mean(hit), 4),
            "Hit@1_CI95": [round(x, 4) for x in hit_ci],
            "Recall@5": round(statistics.mean(r["recall_at_5"] for r in subset), 4),
            "MRR": round(statistics.mean(mrr), 4),
            "MRR_CI95": [round(x, 4) for x in mrr_ci],
            "latency_ms_mean": round(statistics.mean(latencies), 3),
            "latency_ms_p50": round(percentile(latencies, 0.50), 3),
            "latency_ms_p95": round(percentile(latencies, 0.95), 3),
            "latency_ms_p99": round(percentile(latencies, 0.99), 3),
            "errors": sum(r["errors"] > 0 for r in subset),
        }

    category_metrics = {}
    for category in sorted({c["category"] for c in cases}):
        category_metrics[category] = {}
        for strategy in strategies:
            subset = [
                r for r in query_rows
                if r["strategy"] == strategy and r["category"] == category
            ]
            category_metrics[category][strategy] = {
                "Hit@1": round(statistics.mean(r["hit_at_1"] for r in subset), 4),
                "Recall@5": round(statistics.mean(r["recall_at_5"] for r in subset), 4),
                "MRR": round(statistics.mean(r["mrr"] for r in subset), 4),
            }

    failures = []
    for row in query_rows:
        if row["errors"] or row["rank"] is None or row["rank"] > 1:
            failures.append({
                "id": row["id"],
                "category": row["category"],
                "strategy": row["strategy"],
                "question": row["question"],
                "failure_type": (
                    "runtime_error" if row["errors"]
                    else ("miss_at_5" if row["rank"] is None else "not_rank_1")
                ),
                "rank": row["rank"],
                "errors": row["errors"],
            })

    report = {
        "dataset": {
            "path": DATASET.name,
            "cases": len(cases),
            "categories": sorted({c["category"] for c in cases}),
        },
        "benchmark": {
            "repeats_per_query": REPEATS,
            "reranker_cold_start_ms": reranker_cold_start_ms,
            "reranker_warmup_ms": reranker_warmup_ms,
            "latency_aggregation": "median per query, then mean/P50/P95/P99 across queries",
        },
        "rrf": {"k": RRF_K, "confidence_weight": CONFIDENCE_WEIGHT},
        "summary": summary,
        "category_metrics": category_metrics,
        "failure_analysis": failures,
        "query_rows": query_rows,
        "raw_rows": raw_rows,
        "notes": [
            "Benchmark excludes Knowledge Graph retrieval by design.",
            "Reranker cold-start is measured separately from warm inference.",
            "Each query/strategy is repeated three times; median latency is used per query.",
            "Confidence-aware RRF is evaluated against standard RRF rather than against the KG pipeline.",
        ],
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    print("WEATHER HYBRID RETRIEVAL BENCHMARK")
    print("=" * 100)
    print(
        f"Dataset: {len(cases)} queries x {REPEATS} repeats | "
        f"Categories: {', '.join(report['dataset']['categories'])}"
    )
    print(f"Reranker cold start: {reranker_cold_start_ms:.1f} ms | warm-up: {reranker_warmup_ms:.1f} ms")
    print("-" * 100)
    print(f"{'Strategy':<18}{'Hit@1':>9}{'Recall@5':>11}{'MRR':>9}{'Mean ms':>11}{'P50 ms':>11}{'P95 ms':>11}{'P99 ms':>11}")
    print("-" * 100)
    for name in strategies:
        s = summary[name]
        print(
            f"{name:<18}{s['Hit@1']:>9.3f}{s['Recall@5']:>11.3f}{s['MRR']:>9.3f}"
            f"{s['latency_ms_mean']:>11.1f}{s['latency_ms_p50']:>11.1f}"
            f"{s['latency_ms_p95']:>11.1f}{s['latency_ms_p99']:>11.1f}"
        )
    print(f"\nFailure analysis: {len(failures)} non-perfect query/strategy outcomes")
    print(f"Report: {REPORT}")


def _standard_rrf(result_sets: list[list[dict]], top_k: int) -> list[dict]:
    fused: dict[str, dict] = {}
    for results in result_sets:
        for rank, row in enumerate(results, 1):
            key = str(row.get("id"))
            if not key or key == "None":
                continue
            item = fused.setdefault(key, {**row, "rrf_score": 0.0})
            item["rrf_score"] += 1.0 / (RRF_K + rank)
    return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)[:top_k]


if __name__ == "__main__":
    main()
