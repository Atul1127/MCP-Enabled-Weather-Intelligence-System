from rag.citations.validator import validate


def test_unknown_citations_are_removed_without_fabricating_sources():
    answer, sources = validate(
        "Rain is expected [S9].",
        [{"citation": "S1", "source": "weather knowledge"}],
    )
    assert answer == "Rain is expected."
    assert sources == []


def test_valid_citations_are_preserved_and_sources_are_aligned():
    answer, sources = validate(
        "Rainfall is associated with thunderstorms [S1].",
        [{"citation": "S1", "source": "weather knowledge"}],
    )
    assert "[S1]" in answer
    assert sources == [{"citation": "S1", "source": "weather knowledge"}]


def test_no_citation_is_not_invented_for_live_only_answer():
    answer, sources = validate(
        "Current temperature is 29 C.",
        [{"citation": "S1", "source": "weather knowledge"}],
    )
    assert answer == "Current temperature is 29 C."
    assert sources == []
