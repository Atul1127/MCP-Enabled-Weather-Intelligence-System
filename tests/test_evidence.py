from rag.citations.validator import validate
from weather_agent_core.evidence import RAGEvidence, normalize_tool_result
from weather_agent_core.state import AgentState


def test_invalid_citations_are_removed_and_valid_source_is_kept():
    answer, sources = validate("Rain is likely [S9].", [{"citation": "S1", "source": "weather-kb"}])
    assert "S9" not in answer
    assert "[S1]" in answer
    assert sources[0]["citation"] == "S1"


def test_rag_tool_result_becomes_one_evidence_object():
    evidence = normalize_tool_result("search_weather", {"success": True, "query": "rain", "context": "[S1] rain", "documents": [], "sources": [{"citation": "S1"}]})
    assert len(evidence) == 1
    assert isinstance(evidence[0], RAGEvidence)
    assert evidence[0].data["sources"][0]["citation"] == "S1"


def test_required_tool_groups_allow_alternatives():
    state = AgentState(query="forecast", trace_id="test", required_tool_groups=[{"get_forecast", "get_weather"}])
    state.add_observation("get_forecast", {"location": "Kolkata"}, {"success": True})
    assert state.required_requirements_satisfied
