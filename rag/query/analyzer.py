"""Provider-independent query analysis for the RAG pipeline."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryPlan:
    query: str
    intent: str


def analyze(query: str) -> QueryPlan:
    """Classify a retrieval query; the intent is reported alongside results."""
    text = query.strip()
    lower = text.lower()
    comparison = any(x in lower for x in ("compare", "versus", " vs ", "between ", "difference"))
    current = any(x in lower for x in (
        "current weather", "current conditions", "weather right now", "weather now",
        "currently", "at the moment", "at present", "right at this moment",
    ))
    live = any(x in lower for x in ("today", "tomorrow", "now", "forecast", "tonight", "this evening"))
    risk = any(x in lower for x in ("safe", "risk", "suitable", "hike", "run", "play", "outdoor", "travel"))
    intent = "comparison" if comparison else "activity_risk" if risk else "live_weather" if live or current else "knowledge"
    return QueryPlan(query=text, intent=intent)
