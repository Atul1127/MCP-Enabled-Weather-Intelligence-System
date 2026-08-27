"""Build an explicit, auditable execution plan from the routed intent."""
from __future__ import annotations

from .router import classify


class Planner:
    """Create capability-level plans without coupling planning to MCP."""

    def build(self, query: str) -> dict:
        intent = classify(query)
        if intent == "knowledge":
            steps = [{"capability": "knowledge", "preferred_tools": ["search_weather", "ask_weather"]}]
        elif intent == "activity_risk":
            steps = [{"capability": "risk", "preferred_tools": ["assess_weather_risk"]},
                     {"capability": "alerts", "preferred_tools": ["get_weather_alerts"]}]
        elif intent == "comparison":
            steps = [{"capability": "comparison_evidence", "preferred_tools": ["get_forecast", "get_weather", "assess_weather_risk"]},
                     {"capability": "knowledge", "preferred_tools": ["search_weather"]}]
        else:
            steps = [{"capability": "live_weather", "preferred_tools": ["get_weather", "get_forecast", "get_weather_alerts"]},
                     {"capability": "risk", "preferred_tools": ["assess_weather_risk"]}]
        return {"intent": intent, "steps": steps}
