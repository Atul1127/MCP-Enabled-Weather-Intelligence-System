"""Architecture-level contracts that must hold before integration tests."""

from weather_agent_core.planner import Planner
from weather_agent_core.router import classify


def test_router_and_planner_share_intent_contract():
    planner = Planner()
    for query in (
        "weather in Kolkata",
        "forecast for tomorrow in Delhi",
        "compare weather in Mumbai and Pune",
        "is it safe to hike today?",
        "why does humidity feel higher?",
    ):
        assert planner.build(query)["intent"] == classify(query)


def test_comparison_requires_live_and_knowledge_evidence():
    plan = Planner().build("compare weather in Mumbai and Pune")
    assert plan["intent"] == "comparison"
    assert plan["requires_live_data"] is True
    assert plan["requires_knowledge"] is True
    capabilities = {step["capability"] for step in plan["steps"]}
    assert "comparison_evidence" in capabilities
    assert "knowledge" in capabilities
