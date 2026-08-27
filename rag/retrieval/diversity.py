"""Lightweight diversity selection for hybrid retrieval results."""
from __future__ import annotations

import re
from typing import Any


def _tokens(row: dict[str, Any]) -> set[str]:
    text = str(
        row.get("text") or row.get("narrative_text") or row.get("chunk_text") or ""
    ).lower()
    return set(re.findall(r"[a-z0-9]+", text))


def _id(row: dict[str, Any]) -> str | None:
    value = row.get("id", row.get("document_id"))
    return None if value is None else str(value)


def select_mmr(
    candidates: list[dict[str, Any]],
    top_k: int,
    *,
    lambda_mult: float = 0.75,
) -> list[dict[str, Any]]:
    """Select high-scoring but non-redundant candidates.

    Relevance comes from the reranker when available, otherwise hybrid fusion.
    Redundancy is token-set Jaccard similarity, keeping the selector local and
    deterministic without another embedding-model call.
    """
    if top_k < 1 or not candidates:
        return []
    if not 0.0 <= lambda_mult <= 1.0:
        raise ValueError("lambda_mult must be between 0.0 and 1.0")

    pool = list(candidates)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    while pool and len(selected) < top_k:
        best_index: int | None = None
        best_value = float("-inf")

        for i, row in enumerate(pool):
            key = _id(row)
            if key is not None and key in seen:
                continue

            reranker_score = row.get("reranker_score")
            relevance = float(
                reranker_score
                if reranker_score is not None
                else row.get("fusion_score", 0.0) or 0.0
            )
            current = _tokens(row)
            redundancy = 0.0
            if selected and current:
                redundancy = max(
                    len(current & _tokens(other)) / len(current | _tokens(other))
                    for other in selected
                    if current | _tokens(other)
                )

            value = lambda_mult * relevance - (1.0 - lambda_mult) * redundancy
            if value > best_value:
                best_value = value
                best_index = i

        # Remaining candidates may all be duplicates of already-selected IDs.
        if best_index is None:
            break

        chosen = pool.pop(best_index)
        selected.append(chosen)
        key = _id(chosen)
        if key is not None:
            seen.add(key)

    return selected
