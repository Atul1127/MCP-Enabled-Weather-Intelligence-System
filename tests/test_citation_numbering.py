"""Two searches in one request must not produce clashing citation IDs."""
from rag.citations.validator import validate
from weather_agent_core.state import AgentState


def _search(topic_a, topic_b):
    return {
        "success": True,
        "query": topic_a,
        "context": f"[S1] Topic={topic_a}\\ntext a\\n\\n---\\n\\n[S2] Topic={topic_b}\\ntext b",
        "documents": [],
        "sources": [
            {"citation": "S1", "id": topic_a, "source": topic_a},
            {"citation": "S2", "id": topic_b, "source": topic_b},
        ],
    }


def test_second_search_continues_numbering():
    state = AgentState(query="q", trace_id="t")
    first = _search("heat", "humidity")
    second = _search("thunderstorm", "lightning")
    state.add_observation("search_weather", {"query": "a"}, first)
    state.add_observation("search_weather", {"query": "b"}, second)

    assert [source["citation"] for source in state.sources] == ["S1", "S2", "S3", "S4"]
    assert "[S3] Topic=thunderstorm" in second["context"]
    assert "[S4] Topic=lightning" in second["context"]
    assert "[S1] Topic=heat" in first["context"]


def test_validator_maps_each_citation_to_the_right_source():
    state = AgentState(query="q", trace_id="t")
    state.add_observation("search_weather", {"query": "a"}, _search("heat", "humidity"))
    state.add_observation("search_weather", {"query": "b"}, _search("thunderstorm", "lightning"))

    answer, cited = validate("Heat is risky [S1]. Lightning too [S4].", state.sources)
    assert [source["source"] for source in cited] == ["heat", "lightning"]
    assert "[S4] lightning" in answer


def test_single_search_is_unchanged():
    state = AgentState(query="q", trace_id="t")
    result = _search("heat", "humidity")
    state.add_observation("search_weather", {"query": "a"}, result)
    assert [source["citation"] for source in state.sources] == ["S1", "S2"]
    assert result["context"].startswith("[S1]")
