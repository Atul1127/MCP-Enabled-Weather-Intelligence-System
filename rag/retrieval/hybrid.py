"""Hybrid fusion using confidence-aware reciprocal rank fusion."""
from __future__ import annotations
from typing import Any
from advanced_rag import confidence_aware_rrf


def fuse(dense_results: list[dict[str, Any]], sparse_results: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    return confidence_aware_rrf([("dense", dense_results), ("bm25", sparse_results)], top_k)
