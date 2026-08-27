"""Deterministic retrieval metrics used by the RAG evaluation harness."""
from __future__ import annotations
import math
from typing import Callable, Sequence


def recall_at_k(relevances: Sequence[int], k: int) -> float:
    """Binary recall for a ranked list when the denominator is known relevant items."""
    if k < 1:
        raise ValueError("k must be positive")
    total = sum(1 for value in relevances if value)
    if total == 0:
        return 0.0
    return min(1.0, sum(1 for value in relevances[:k] if value) / total)


def precision_at_k(relevances: Sequence[int], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    window = list(relevances[:k])
    return sum(1 for value in window if value) / k


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


def topic_relevance(
    documents: Sequence[dict], expected_topics: Sequence[str], *, field: str = "topic"
) -> list[int]:
    """Mark a ranked document relevant when its topic matches any expected topic.

    This is intentionally a topic-proxy metric. True document-level Recall@K
    requires explicit relevant document IDs in the evaluation dataset.
    """
    topics = [str(topic).strip().lower() for topic in expected_topics if str(topic).strip()]
    relevance: list[int] = []
    for document in documents:
        value = str(document.get(field) or "").lower()
        relevance.append(int(any(topic in value or value in topic for topic in topics)))
    return relevance


def evaluate_ranked_documents(
    documents: Sequence[dict], expected_topics: Sequence[str], ks: Sequence[int] = (1, 3, 5, 10)
) -> dict[str, float]:
    relevance = topic_relevance(documents, expected_topics)
    # For topic-proxy recall, one hit is sufficient evidence that the expected
    # topic was retrieved; document-level recall is reported separately once
    # explicit relevant IDs are available.
    first_hit = reciprocal_rank(relevance)
    result: dict[str, float] = {"mrr": first_hit, "topic_hit": float(any(relevance))}
    for k in ks:
        result[f"precision_at_{k}"] = precision_at_k(relevance, k)
        result[f"ndcg_at_{k}"] = ndcg_at_k(relevance, k)
        result[f"topic_recall_at_{k}"] = float(any(relevance[:k]))
    return result
