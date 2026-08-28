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
        knowledge = intent == "knowledge"
        live = intent in {"live_weather", "activity_risk", "comparison"}

        if knowledge:
            # RAG benchmark contract: both retrieval capabilities are required.
            # Keep them as separate required steps so state/verifier accounting
            # cannot accidentally treat the pair as an OR choice.
            steps = (
                PlanStep(
                    "knowledge_search",
                    "knowledge_search",
                    ("search_weather",),
                    required=True,
                    parallelizable=False,
                ),
                PlanStep(
                    "knowledge_answer",
                    "knowledge",
                    ("ask_weather",),
                    required=True,
                    parallelizable=False,
                ),
            )
        elif risk:
            steps = (
                PlanStep("risk", "risk", ("assess_weather_risk",)),
                PlanStep("alerts", "alerts", ("get_weather_alerts",), required=False),
            )
        elif comparison:
            steps = (
                PlanStep(
                    "comparison_evidence",
                    "comparison_evidence",
                    ("get_forecast", "get_weather", "assess_weather_risk"),
                ),
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
