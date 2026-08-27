"""Lightweight diversity selection for hybrid retrieval results."""
from __future__ import annotations
import re
from typing import Any


def _tokens(row: dict[str, Any]) -> set[str]:
    text = str(row.get("text") or row.get("narrative_text") or row.get("chunk_text") or "").lower()
    return set(re.findall(r"[a-z0-9]+", text))


def _id(row: dict[str, Any]) -> str | None:
    value = row.get("id", row.get("document_id"))
    return None if value is None else str(value)


def select_mmr(candidates: list[dict[str, Any]], top_k: int, *, lambda_mult: float = 0.75) -> list[dict[str, Any]]:
    """Select high-scoring but non-redundant candidates.

    The relevance score comes from hybrid fusion; redundancy is token-set
    Jaccard similarity. This keeps the selector local and deterministic and
    avoids another embedding model call after reranking.
    """
    if top_k < 1 or not candidates:
        return []
    lam = max(0.0, min(1.0, float(lambda_mult)))
    pool = list(candidates)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    while pool and len(selected) < top_k:
        best_index = 0
        best_value = float("-inf")
        for i, row in enumerate(pool):
            key = _id(row)
            if key is not None and key in seen:
                continue
            relevance = float(row.get("reranker_score") if row.get("reranker_score") is not None else row.get("fusion_score", 0.0) or 0.0)
            current = _tokens(row)
            redundancy = 0.0
            if selected and current:
                redundancy = max(
                    (len(current & _tokens(other)) / len(current | _tokens(other))) if current | _tokens(other) else 0.0
                    for other in selected
                )
            value = lam * relevance - (1.0 - lam) * redundancy
            if value > best_value:
                best_value, best_index = value, i
        chosen = pool.pop(best_index)
        selected.append(chosen)
        key = _id(chosen)
        if key is not None:
            seen.add(key)
    return selected
