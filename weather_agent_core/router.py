"""Deterministic first-pass intent router.

Routing is intentionally cheap and inspectable. Gemini reasons over tool
results; the router decides which capability family must be available.
"""
from __future__ import annotations

import re


LIVE_MARKERS = ("today", "tomorrow", "now", "right now", "tonight", "this evening", "forecast", "next week", "next few days")
COMPARISON_MARKERS = ("compare", "versus", " vs ", "between ", "which is better")
KNOWLEDGE_MARKERS = (
    "typically",
    "usually",
    "what causes",
    "why does",
    "how does",
    "what is",
    "what are",
    "associated with",
    "meaning of",
    "mean",
    "means",
    "does ... mean",
)
RISK_MARKERS = ("safe", "risk", "suitable", "should i", "play", "run", "hike", "travel", "outdoor")


def classify(query: str) -> str:
    text = query.lower().strip()
    if any(marker in text for marker in COMPARISON_MARKERS):
        return "comparison"
    # Risk intent takes precedence over generic future/live markers because
    # questions such as "is cricket safe tomorrow?" need the risk tool.
    if any(marker in text for marker in RISK_MARKERS):
        return "activity_risk"
    # Conceptual/definition questions should use the knowledge/RAG path.
    # Avoid broadening this into every "what is" query when the user clearly
    # asks for live weather.
    if any(marker in text for marker in KNOWLEDGE_MARKERS) and not any(marker in text for marker in LIVE_MARKERS):
        return "knowledge"
    if any(marker in text for marker in LIVE_MARKERS):
        return "live_weather"
    return "weather_intelligence"


def is_simple_current(query: str) -> bool:
    text = query.lower().strip()
    if any(marker in text for marker in LIVE_MARKERS if marker != "now"):
        return False
    patterns = (r"\bcurrent weather\b", r"\bcurrent conditions?\b", r"\bweather right now\b", r"\bweather now\b")
    return any(re.search(pattern, text) for pattern in patterns)
