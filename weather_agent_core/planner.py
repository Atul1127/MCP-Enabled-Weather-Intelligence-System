"""Build explicit, auditable execution plans from routed intent."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .router import classify


@dataclass(frozen=True)
class PlanStep:
    id: str
    capability: str
    preferred_tools: tuple[str, ...]
    required: bool = True
    parallelizable: bool = True


@dataclass(frozen=True)
class ExecutionPlan:
    intent: str
    steps: tuple[PlanStep, ...]
    requires_live_data: bool
    requires_knowledge: bool

    def as_dict(self) -> dict:
        return {
            "intent": self.intent,
            "requires_live_data": self.requires_live_data,
            "requires_knowledge": self.requires_knowledge,
            "steps": [asdict(step) for step in self.steps],
        }


class Planner:
    """Create capability-level plans without coupling planning to MCP."""

    def build(self, query: str) -> dict:
        intent = classify(query)
        text = query.lower()
        comparison = intent == "comparison"
        risk = intent == "activity_risk"
        alerts = intent == "alerts"
        knowledge = intent == "knowledge"
        live = intent in {"live_weather", "activity_risk", "comparison", "alerts"}

        if knowledge:
            steps = (
                PlanStep(
                    "knowledge",
                    "knowledge",
                    ("search_weather", "ask_weather"),
                    required=True,
                    parallelizable=False,
                ),
            )
        elif alerts:
            # Alerts are a distinct live capability. Do not make alert lookup
            # optional merely because the query also contains a time phrase such
            # as "this week" or "next few days".
            steps = (
                PlanStep("alerts", "alerts", ("get_weather_alerts",), required=True, parallelizable=False),
            )
        elif risk:
            steps = (
                PlanStep("risk", "risk", ("assess_weather_risk",)),
                PlanStep("alerts", "alerts", ("get_weather_alerts",), required=False),
            )
        elif comparison:
            # Comparison questions need the capability matching the comparison
            # target. Outdoor suitability/cricket/risk language means risk;
            # explicit forecast language means forecast. Keep the knowledge
            # lookup optional so comparisons do not trigger unnecessary RAG work.
            risk_comparison = any(
                marker in text
                for marker in ("outdoor", "cricket", "run", "safe", "risk", "suitable", "activity")
            )
            forecast_comparison = "forecast" in text
            preferred = (
                ("assess_weather_risk",)
                if risk_comparison and not forecast_comparison
                else ("get_forecast", "get_weather")
            )
            steps = (
                PlanStep("comparison_evidence", "comparison_evidence", preferred),
                PlanStep("comparison_knowledge", "knowledge", ("search_weather",), required=False),
            )
        else:
            preferred = (
                ("get_forecast", "get_weather")
                if any(x in text for x in ("forecast", "tomorrow", "next week"))
                else ("get_weather", "get_forecast")
            )
            steps = (
                PlanStep("live_weather", "live_weather", preferred),
                PlanStep("hazards", "alerts", ("get_weather_alerts",), required=False),
            )

        plan = ExecutionPlan(
            intent=intent,
            steps=steps,
            requires_live_data=live,
            requires_knowledge=knowledge or comparison,
        )
        return plan.as_dict()
