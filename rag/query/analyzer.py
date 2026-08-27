"""Provider-independent query analysis for the RAG pipeline."""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class QueryPlan:
    query: str
    intent: str
    complexity: str
    needs_expansion: bool
    filters: dict[str, str | None]


def analyze(query: str, *, location: str | None = None, state: str | None = None) -> QueryPlan:
    text = query.strip()
    lower = text.lower()
    words = re.findall(r"\w+", lower)
    comparison = any(x in lower for x in ("compare", "versus", " vs ", "between ", "difference"))
    live = any(x in lower for x in ("today", "tomorrow", "now", "forecast", "tonight", "this evening"))
    risk = any(x in lower for x in ("safe", "risk", "suitable", "hike", "run", "play", "outdoor", "travel"))
    intent = "comparison" if comparison else "activity_risk" if risk else "live_weather" if live else "knowledge"
    complex_query = comparison or len(words) > 10 or any(x in lower for x in ("why", "how", "factors", "relationship"))
    return QueryPlan(
        query=text,
        intent=intent,
        complexity="complex" if complex_query else "simple",
        needs_expansion=complex_query and intent == "knowledge",
        filters={"location": location, "state": state},
    )
