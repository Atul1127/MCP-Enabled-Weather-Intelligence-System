from evaluation.rag_llm_eval import cited_ids, topic_match


def test_cited_ids_are_sorted_and_unique():
    assert cited_ids("Claim [S2]. Another [S1]. Again [S2].") == ["S1", "S2"]


def test_topic_match_checks_expected_topics():
    sources = [{"topic": "heavy rainfall"}, {"topic": "thunderstorm"}]
    assert topic_match(sources, ["heavy rainfall"])
    assert not topic_match(sources, ["flood"])
