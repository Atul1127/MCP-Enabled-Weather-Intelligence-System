"""Benchmark Dense, BM25, RRF, confidence-aware RRF and reranking.

Run from the repository root:
    python evaluation/retrieval_benchmark.py

The benchmark intentionally excludes the Knowledge Graph. It measures the
non-KG retrieval stack inherited from the original Hybrid-RAG project.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

# When Python executes a file inside evaluation/, it places that directory on
# sys.path rather than the repository root. Add the root explicitly so the
# benchmark works both as `python evaluation/...` and from an IDE/test runner.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advanced_rag import CONFIDENCE_WEIGHT, RRF_K, confidence_aware_rrf, rerank
from local_rag_store import get_store

EVALUATION_DIR = Path(__file__).resolve().parent
DATASET = EVALUATION_DIR / "weather_retrieval_dataset.json"
REPORT = EVALUATION_DIR / "retrieval_benchmark_report.json"


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


def main() -> None:
    cases = json.loads(DATASET.read_text(encoding="utf-8"))["cases"]
    store = get_store()
    strategies = ["Dense", "BM25", "RRF", "ConfidenceRRF", "Reranker"]
    rows: list[dict] = []

    for case in cases:
        q = case["question"]
        dense = None
        bm25 = None
        confidence_fused = None

        for strategy in strategies:
            started = time.perf_counter()
            error = None
            try:
                if strategy == "Dense":
                    result = store.dense_search(q, 5)
                elif strategy == "BM25":
                    result = store.bm25_search(q, 5)
                elif strategy == "RRF":
                    dense = store.dense_search(q, 20)
                    bm25 = store.bm25_search(q, 20)
                    result = _standard_rrf([dense, bm25], 5)
                elif strategy == "ConfidenceRRF":
                    dense = store.dense_search(q, 20)
                    bm25 = store.bm25_search(q, 20)
                    confidence_fused = confidence_aware_rrf(
                        [("dense", dense), ("bm25", bm25)], 20
                    )
                    result = confidence_fused[:5]
                else:
                    if confidence_fused is None:
                        dense = store.dense_search(q, 20)
                        bm25 = store.bm25_search(q, 20)
                        confidence_fused = confidence_aware_rrf(
                            [("dense", dense), ("bm25", bm25)], 20
                        )
                    result = rerank(q, confidence_fused, 5)
            except Exception as exc:
                result = []
                error = f"{type(exc).__name__}: {exc}"

            latency_ms = (time.perf_counter() - started) * 1000
            row = {
                "id": case["id"],
                "category": case["category"],
                "question": q,
                "gold_source": case["gold_source"],
                "strategy": strategy,
                "latency_ms": round(latency_ms, 3),
                "error": error,
            }
            row.update(metrics(result, case["gold_source"]))
            rows.append(row)

    summary = {}
    for strategy in strategies:
        subset = [r for r in rows if r["strategy"] == strategy]
        latencies = sorted(r["latency_ms"] for r in subset)
        p95_index = max(0, min(len(latencies) - 1, int(len(latencies) * 0.95) - 1))
        summary[strategy] = {
            "count": len(subset),
            "Hit@1": round(statistics.mean(r["hit_at_1"] for r in subset), 4),
            "Recall@5": round(statistics.mean(r["recall_at_5"] for r in subset), 4),
            "MRR": round(statistics.mean(r["mrr"] for r in subset), 4),
            "latency_ms_mean": round(statistics.mean(latencies), 3),
            "latency_ms_p50": round(statistics.median(latencies), 3),
            "latency_ms_p95": round(latencies[p95_index], 3),
            "errors": sum(r["error"] is not None for r in subset),
        }

    category_metrics = {}
    for category in sorted({c["category"] for c in cases}):
        category_metrics[category] = {}
        for strategy in strategies:
            subset = [
                r for r in rows
                if r["strategy"] == strategy and r["category"] == category
            ]
            category_metrics[category][strategy] = {
                "Hit@1": round(statistics.mean(r["hit_at_1"] for r in subset), 4),
                "Recall@5": round(statistics.mean(r["recall_at_5"] for r in subset), 4),
                "MRR": round(statistics.mean(r["mrr"] for r in subset), 4),
            }

    failures = []
    for row in rows:
        if row["error"] or row["rank"] is None or row["rank"] > 1:
            failures.append({
                "id": row["id"],
                "category": row["category"],
                "strategy": row["strategy"],
                "question": row["question"],
                "failure_type": (
                    "runtime_error" if row["error"]
                    else ("miss_at_5" if row["rank"] is None else "not_rank_1")
                ),
                "rank": row["rank"],
                "error": row["error"],
            })

    report = {
        "dataset": {
            "path": DATASET.name,
            "cases": len(cases),
            "categories": sorted({c["category"] for c in cases}),
        },
        "rrf": {"k": RRF_K, "confidence_weight": CONFIDENCE_WEIGHT},
        "summary": summary,
        "category_metrics": category_metrics,
        "failure_analysis": failures,
        "rows": rows,
        "notes": [
            "Benchmark excludes Knowledge Graph retrieval by design.",
            "Latency covers retrieval/reranking only and includes first-run model/cache effects.",
            "Use repeated runs for stable latency comparisons.",
        ],
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("WEATHER HYBRID RETRIEVAL BENCHMARK")
    print("=" * 88)
    print(
        f"Dataset: {len(cases)} queries | "
        f"Categories: {', '.join(report['dataset']['categories'])}"
    )
    print(
        f"{'Strategy':<18}{'Hit@1':>9}{'Recall@5':>11}"
        f"{'MRR':>9}{'Mean ms':>11}{'P95 ms':>11}"
    )
    print("-" * 88)
    for name in strategies:
        s = summary[name]
        print(
            f"{name:<18}{s['Hit@1']:>9.3f}{s['Recall@5']:>11.3f}"
            f"{s['MRR']:>9.3f}{s['latency_ms_mean']:>11.1f}"
            f"{s['latency_ms_p95']:>11.1f}"
        )
    print(f"\nFailure analysis: {len(failures)} non-perfect outcomes")
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
    return sorted(
        fused.values(), key=lambda x: x["rrf_score"], reverse=True
    )[:top_k]


if __name__ == "__main__":
    main()
