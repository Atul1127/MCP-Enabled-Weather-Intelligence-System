"""Code-computed confidence and the numeric grounding check."""
from weather_agent_core.answer_checks import evidence_status, unverified_numbers

LIVE_PLAN = {"requires_live_data": True, "requires_knowledge": False}


def test_complete_when_live_data_succeeds():
    obs = [{"tool": "get_forecast", "result": {"success": True}}]
    assert evidence_status(LIVE_PLAN, obs, {"sufficient": True}) == {"level": "complete", "reasons": []}


def test_no_live_data_when_every_live_tool_fails():
    obs = [{"tool": "get_forecast", "result": {"success": False, "error": "x"}}]
    assert evidence_status(LIVE_PLAN, obs, {"sufficient": False})["level"] == "no_live_data"


def test_partial_when_an_optional_tool_fails():
    obs = [
        {"tool": "get_forecast", "result": {"success": True}},
        {"tool": "get_weather_alerts", "result": {"success": False}},
    ]
    status = evidence_status(LIVE_PLAN, obs, {"sufficient": True})
    assert status["level"] == "partial"
    assert "get_weather_alerts" in status["reasons"][0]


def test_partial_when_knowledge_search_is_empty():
    plan = {"requires_live_data": False, "requires_knowledge": True}
    obs = [{"tool": "search_weather", "result": {"success": True, "documents": []}}]
    assert evidence_status(plan, obs, {"sufficient": True})["level"] == "partial"


def test_numbers_present_in_evidence_pass_with_rounding():
    evidence = [{"data": {"temperature_c": 30.8, "precipitation_probability_pct": 70, "max_wind_kmh": 18.4}}]
    answer = "Expect about 31°C, a 70% chance of rain and winds near 18 km/h [S1]."
    assert unverified_numbers(answer, evidence) == []


def test_invented_numbers_are_reported():
    evidence = [{"data": {"temperature_c": 30.8, "precipitation_probability_pct": 70}}]
    answer = "It will be 36°C with 25 mm of rain and a 70% chance of showers."
    assert unverified_numbers(answer, evidence) == ["36°C", "25 mm"]


def test_numbers_quoted_from_knowledge_text_count_as_evidence():
    evidence = [{"data": {"context": "[S1] IMD calls 64.5 mm or more in 24 hours heavy rainfall."}}]
    assert unverified_numbers("Heavy rain means at least 64.5 mm in a day [S1].", evidence) == []
