from advanced_rag import compress_context, confidence_aware_rrf, rrf_merge


def test_rrf_fuses_dense_and_sparse_results():
    dense = [{"id": "a", "text": "rain", "dense_score": 0.9}, {"id": "b", "text": "wind", "dense_score": 0.8}]
    sparse = [{"id": "b", "text": "wind", "bm25_score": 4.0}, {"id": "c", "text": "heat", "bm25_score": 3.0}]
    result = rrf_merge([dense, sparse], top_k=3)
    assert [row["id"] for row in result] == ["b", "a", "c"]
    assert result[0]["rrf_score"] > result[1]["rrf_score"]


def test_confidence_aware_rrf_records_confidence_and_provenance():
    dense = [{"id": "a", "dense_score": 0.95}, {"id": "b", "dense_score": 0.70}]
    bm25 = [{"id": "b", "bm25_score": 9.0}, {"id": "a", "bm25_score": 1.0}]
    result = confidence_aware_rrf([("dense", dense), ("bm25", bm25)], top_k=2)
    assert {row["id"] for row in result} == {"a", "b"}
    assert all("fusion_score" in row for row in result)
    assert all(0.0 <= row["retrieval_confidence"] <= 1.0 for row in result)
    assert result[0]["retrieval_channels"]
    assert result[0]["retrieval_ranks"]


def test_context_compression_keeps_citations():
    context, sources = compress_context(
        "heavy rainfall thunderstorm",
        [
            {
                "id": "rain-1",
                "title": "Rain",
                "topic": "rainfall",
                "source": "test",
                "text": "Heavy rainfall can occur with thunderstorms. Clear skies are different.",
                "reranker_score": 1.2,
            }
        ],
        max_chars=1000,
    )
    assert "[S1]" in context
    assert "Heavy rainfall" in context
    assert sources[0]["id"] == "rain-1"
