"""Hybrid retrieval fusion using confidence-aware reciprocal rank fusion."""
from __future__ import annotations
from typing import Any
import os

RRF_K = int(os.environ.get("WEATHER_RRF_K", "60"))
CONFIDENCE_WEIGHT = float(os.environ.get("WEATHER_RRF_CONFIDENCE_WEIGHT", "0.5"))

def _confidence(values: list[float], value: float) -> float:
    if not values:
        return 0.0
    lo, hi = min(values), max(values)
    if hi <= lo:
        return 1.0 if value > 0 else 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))

def _score(row: dict[str, Any], key: str, fallback: str) -> float:
    try:
        return float(row.get(key, row.get(fallback, 0.0)) or 0.0)
    except (TypeError, ValueError):
        return 0.0

def _id(row: dict[str, Any]) -> str | None:
    value = row.get("id", row.get("document_id"))
    if value is None or str(value).strip() in {"", "None"}:
        return None
    return str(value)

def fuse(dense_results: list[dict[str, Any]], sparse_results: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if top_k < 1:
        return []
    fused: dict[str, dict[str, Any]] = {}
    for channel, results, key_name, fallback in (("dense", dense_results, "dense_score", "similarity"), ("bm25", sparse_results, "bm25_score", "score")):
        values = [_score(r, key_name, fallback) for r in results]
        for rank, row in enumerate(results, 1):
            key = _id(row)
            if key is None:
                continue
            score = _score(row, key_name, fallback)
            confidence = _confidence(values, score)
            item = fused.setdefault(key, {**row, "id": key, "rrf_score": 0.0, "confidence_score": 0.0, "retrieval_ranks": {}, "retrieval_channels": []})
            item["rrf_score"] += 1 / (RRF_K + rank)
            item["confidence_score"] += CONFIDENCE_WEIGHT * confidence / (RRF_K + rank)
            item["retrieval_ranks"][channel] = rank
            item["retrieval_channels"].append(channel)
            item["retrieval_confidence"] = max(float(item.get("retrieval_confidence", 0.0)), confidence)
    for item in fused.values():
        item["fusion_score"] = item["rrf_score"] + item["confidence_score"]
    return sorted(fused.values(), key=lambda x: x["fusion_score"], reverse=True)[:top_k]
