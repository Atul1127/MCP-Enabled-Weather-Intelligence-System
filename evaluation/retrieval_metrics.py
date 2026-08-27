"""Deterministic retrieval metrics for topic-proxy and document-level evaluation."""
from __future__ import annotations

import math
from typing import Sequence


def recall_at_k(relevances: Sequence[int], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    total = sum(1 for value in relevances if value)
    if total == 0:
        return 0.0
    return min(1.0, sum(1 for value in relevances[:k] if value) / total)


def precision_at_k(relevances: Sequence[int], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    return sum(1 for value in relevances[:k] if value) / k


def reciprocal_rank(relevances: Sequence[int]) -> float:
    for rank, value in enumerate(relevances, 1):
        if value:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevances: Sequence[int], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    actual = list(relevances[:k])
    dcg = sum(value / math.log2(rank + 1) for rank, value in enumerate(actual, 1))
    ideal = sorted((int(bool(value)) for value in relevances), reverse=True)[:k]
    idcg = sum(value / math.log2(rank + 1) for rank, value in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def document_relevance(documents: Sequence[dict], relevant_ids: Sequence[str]) -> list[int]:
    """Mark each unique ranked document as relevant at most once.

    Duplicate IDs remain as non-relevant rank positions so rank-based metrics
    retain their original positions, while recall cannot count one document twice.
    """
    expected = {str(value).strip() for value in relevant_ids if str(value).strip()}
    seen: set[str] = set()
    relevance: list[int] = []
    for doc in documents:
        doc_id = str(doc.get("id", doc.get("document_id", ""))).strip()
        if doc_id and doc_id in seen:
            relevance.append(0)
            continue
        if doc_id:
            seen.add(doc_id)
        relevance.append(int(bool(doc_id and doc_id in expected)))
    return relevance


def evaluate_relevant_documents(
    documents: Sequence[dict], relevant_ids: Sequence[str], ks: Sequence[int] = (1, 3, 5, 10)
) -> dict[str, float]:
    """Evaluate a ranked list against canonical relevant document IDs."""
    expected = {str(value).strip() for value in relevant_ids if str(value).strip()}
    relevance = document_relevance(documents, expected)
    result: dict[str, float] = {"mrr": reciprocal_rank(relevance)}
    for k in ks:
        result[f"precision_at_{k}"] = precision_at_k(relevance, k)
        result[f"recall_at_{k}"] = (
            min(1.0, sum(relevance[:k]) / len(expected)) if expected else 0.0
        )
        result[f"ndcg_at_{k}"] = ndcg_at_k(relevance, k)
    return result


def topic_relevance(
    documents: Sequence[dict], expected_topics: Sequence[str], *, field: str = "topic"
) -> list[int]:
    topics = [str(topic).strip().lower() for topic in expected_topics if str(topic).strip()]
    return [
        int(
            any(
                topic in str(document.get(field) or "").lower()
                or str(document.get(field) or "").lower() in topic
                for topic in topics
            )
        )
        for document in documents
    ]


def evaluate_ranked_documents(
    documents: Sequence[dict], expected_topics: Sequence[str], ks: Sequence[int] = (1, 3, 5, 10)
) -> dict[str, float]:
    relevance = topic_relevance(documents, expected_topics)
    result: dict[str, float] = {"mrr": reciprocal_rank(relevance), "topic_hit": float(any(relevance))}
    for k in ks:
        result[f"precision_at_{k}"] = precision_at_k(relevance, k)
        result[f"ndcg_at_{k}"] = ndcg_at_k(relevance, k)
        result[f"topic_recall_at_{k}"] = float(any(relevance[:k]))
    return result
