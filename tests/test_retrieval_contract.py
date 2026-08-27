"""Regression tests for retrieval fusion invariants."""

from rag.retrieval.hybrid import fuse


def test_fusion_ignores_rows_without_ids():
    results = fuse(
        [{"dense_score": 1.0}, {"id": "valid", "dense_score": 0.5}],
        [{"id": "valid", "bm25_score": 2.0}],
        top_k=10,
    )
    assert [row["id"] for row in results] == ["valid"]


def test_fusion_respects_top_k():
    dense = [{"id": str(i), "dense_score": float(10 - i)} for i in range(10)]
    results = fuse(dense, [], top_k=3)
    assert len(results) == 3
