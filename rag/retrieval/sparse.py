"""Sparse BM25 retrieval adapter."""
from __future__ import annotations
from typing import Any


def search(store: Any, query: str, top_k: int, allowed: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    return store.bm25_search(query, top_k, allowed)
