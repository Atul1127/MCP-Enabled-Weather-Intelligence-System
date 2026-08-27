"""Contract tests for the modular RAG stack."""

from rag.pipeline import RAGPipeline
from rag.retrieval.hybrid import fuse


def test_hybrid_fusion_prefers_cross_channel_evidence():
    dense = [{"id": "a", "dense_score": 0.9, "text": "rain"}, {"id": "b", "dense_score": 0.8, "text": "wind"}]
    sparse = [{"id": "b", "bm25_score": 5.0, "text": "wind"}, {"id": "c", "bm25_score": 4.0, "text": "storm"}]
    results = fuse(dense, sparse, top_k=3)
    assert results
    assert results[0]["id"] == "b"
    assert set(results[0]["retrieval_channels"]) == {"dense", "bm25"}


def test_pipeline_can_be_constructed_without_llm_side_effects():
    assert RAGPipeline() is not None
