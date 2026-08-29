from weather_agent_core.planner import Planner
from weather_agent_core.router import classify


def test_weather_alert_queries_route_to_alerts():
    queries = (
        "Are there any weather hazards for Kolkata over the next few days?",
        "Check Mumbai for actionable weather hazards this week.",
    )
    for query in queries:
        assert classify(query) == "alerts"
        plan = Planner().build(query)
        required = [step for step in plan["steps"] if step.get("required", True)]
        assert len(required) == 1
        assert required[0]["preferred_tools"] == ("get_weather_alerts",)
        assert plan["requires_live_data"] is True
