from rag.context.compressor import compress
from rag.retrieval.hybrid import fuse


def test_rrf_combines_dense_and_bm25_results():
    dense = [{"document_id": "doc1", "similarity": 0.90}, {"document_id": "doc2", "similarity": 0.80}]
    sparse = [{"document_id": "doc2", "bm25_score": 8.0}, {"document_id": "doc3", "bm25_score": 7.0}]

    results = fuse(dense, sparse, top_k=3)

    assert len(results) == 3
    ids = {result["id"] for result in results}
    assert ids == {"doc1", "doc2", "doc3"}
    shared = next(result for result in results if result["id"] == "doc2")
    assert set(shared["retrieval_channels"]) == {"dense", "bm25"}
    assert shared["rrf_score"] > 0


def test_shared_document_gets_score_from_both_rankers():
    result = fuse([{"document_id": "doc1"}], [{"document_id": "doc1"}], top_k=1)[0]
    assert result["id"] == "doc1"
    assert result["retrieval_ranks"] == {"dense": 1, "bm25": 1}
    assert result["rrf_score"] > 0


def test_rrf_respects_top_k():
    dense = [{"document_id": f"doc{i}"} for i in range(1, 5)]
    results = fuse(dense, [], top_k=2)
    assert len(results) == 2


def test_build_context_creates_citations():
    documents = [{
        "document_id": "doc1",
        "location": "Kolkata",
        "state": "West Bengal",
        "source": "open-meteo",
        "source_type": "forecast",
        "headline": "Weather forecast for Kolkata",
        "chunk_text": "Rain is expected.",
        "similarity": 0.82,
        "bm25_score": 4.5,
    }]

    context, sources = compress("rain in Kolkata", documents)

    assert "[S1]" in context
    assert "Kolkata" in context
    assert "Rain is expected." in context
    assert len(sources) == 1
    assert sources[0]["citation"] == "S1"
    assert sources[0]["document_id"] == "doc1"
