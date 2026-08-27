"""Cross-encoder reranking adapter."""
from __future__ import annotations
from typing import Any
from advanced_rag import rerank as _rerank


def rerank(query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    return _rerank(query, candidates, top_k)
