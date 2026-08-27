"""Benchmark the modular local weather retrieval stack."""
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
from local_rag_store import get_store
from rag.retrieval.dense import search as dense_search
from rag.retrieval.sparse import search as sparse_search
from rag.retrieval.hybrid import fuse
from rag.reranking.cross_encoder import rerank

DATASET = Path(__file__).resolve().parent / "weather_retrieval_dataset.json"
REPORT = Path(__file__).resolve().parent / "retrieval_benchmark_report.json"

def metrics(rows: list[dict], gold: str) -> dict:
    rank = next((i for i, row in enumerate(rows, 1) if str(row.get("id")) == gold), None)
    return {"rank": rank, "hit_at_1": int(rank == 1), "recall_at_5": int(rank is not None and rank <= 5), "mrr": 0.0 if rank is None else 1.0 / rank}

def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    position = (len(values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)

def main() -> None:
    cases = json.loads(DATASET.read_text(encoding="utf-8"))["cases"]
    store = get_store()
    strategies = {"dense": [], "bm25": [], "hybrid": [], "reranked": []}
    rows = []
    for case in cases:
        query, gold = case["question"], case["gold_source"]
        started = time.perf_counter(); dense = dense_search(store, query, 10); dense_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter(); sparse = sparse_search(store, query, 10); sparse_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter(); hybrid = fuse(dense, sparse, 10); hybrid_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter(); reranked = rerank(query, hybrid, min(5, len(hybrid))); rerank_ms = (time.perf_counter() - started) * 1000
        for name, result, latency in (("dense", dense, dense_ms), ("bm25", sparse, sparse_ms), ("hybrid", hybrid, hybrid_ms), ("reranked", reranked, rerank_ms)):
            value = metrics(result, gold); value.update({"id": case["id"], "category": case["category"], "latency_ms": latency}); strategies[name].append(value)
        rows.append({"id": case["id"], "category": case["category"], "gold_source": gold, "dense": metrics(dense, gold), "bm25": metrics(sparse, gold), "hybrid": metrics(hybrid, gold), "reranked": metrics(reranked, gold)})
    summary = {}
    for name, values in strategies.items():
        summary[name] = {"cases": len(values), "hit_at_1": statistics.mean(v["hit_at_1"] for v in values), "recall_at_5": statistics.mean(v["recall_at_5"] for v in values), "mrr": statistics.mean(v["mrr"] for v in values), "mean_latency_ms": statistics.mean(v["latency_ms"] for v in values), "p95_latency_ms": percentile([v["latency_ms"] for v in values], 0.95)}
    REPORT.write_text(json.dumps({"dataset": DATASET.name, "summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
