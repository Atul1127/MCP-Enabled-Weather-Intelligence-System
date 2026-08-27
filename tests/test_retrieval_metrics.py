import pytest

from evaluation.retrieval_metrics import (
    evaluate_ranked_documents,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    topic_relevance,
)


def test_topic_relevance_uses_ranked_topics():
    docs = [{"topic": "forecast"}, {"topic": "heat"}, {"topic": "forecast uncertainty"}]
    assert topic_relevance(docs, ["forecast"]) == [1, 0, 1]


def test_precision_and_reciprocal_rank():
    relevance = [0, 1, 1]
    assert precision_at_k(relevance, 2) == pytest.approx(0.5)
    assert reciprocal_rank(relevance) == pytest.approx(0.5)


def test_ndcg_is_bounded():
    score = ndcg_at_k([1, 0, 1], 3)
    assert 0.0 <= score <= 1.0


def test_evaluate_ranked_documents_reports_topic_metrics():
    metrics = evaluate_ranked_documents(
        [{"topic": "heat"}, {"topic": "forecast"}], ["forecast"], (1, 2)
    )
    assert metrics["topic_recall_at_1"] == 0.0
    assert metrics["topic_recall_at_2"] == 1.0
    assert metrics["mrr"] == pytest.approx(0.5)
