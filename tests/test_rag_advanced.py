from rag.query.expansion import expand, parse
from rag.retrieval.diversity import select_mmr


def test_query_expansion_preserves_original_and_dedupes_at_pipeline_boundary():
    raw = '{"queries":["rain risk Kolkata", "rain risk Kolkata", "outdoor weather Kolkata"]}'
    assert parse(raw) == ["rain risk Kolkata", "rain risk Kolkata", "outdoor weather Kolkata"]
    assert expand("compare long weather knowledge for Kolkata and Mumbai conditions", lambda _: raw, max_variants=2) == [
        "compare long weather knowledge for Kolkata and Mumbai conditions",
        "rain risk Kolkata",
        "rain risk Kolkata",
    ]


def test_mmr_removes_duplicate_ids_and_prefers_diverse_evidence():
    candidates = [
        {"id": "a", "text": "heavy rain Kolkata forecast", "reranker_score": 0.99},
        {"id": "a", "text": "heavy rain Kolkata forecast duplicate", "reranker_score": 0.98},
        {"id": "b", "text": "strong wind Mumbai forecast", "reranker_score": 0.90},
    ]
    selected = select_mmr(candidates, 2)
    assert [row["id"] for row in selected] == ["a", "b"]


def test_mmr_validates_lambda():
    assert select_mmr([{"id": "a", "text": "x", "fusion_score": 1.0}], 1, lambda_mult=0.0)[0]["id"] == "a"
