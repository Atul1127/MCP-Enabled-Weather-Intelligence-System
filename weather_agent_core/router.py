"""Deterministic first-pass intent router.

Routing is intentionally cheap and inspectable. Gemini reasons over tool
results; the router decides which capability family must be available.
"""
from __future__ import annotations

import re


LIVE_MARKERS = ("today", "tomorrow", "now", "right now", "tonight", "this evening", "forecast", "next week", "next few days")
COMPARISON_MARKERS = ("compare", "versus", " vs ", "between ", "which is better")
ALERT_MARKERS = (
    "weather alert", "weather alerts", "weather hazard", "weather hazards",
    "dangerous weather", "actionable weather hazard", "actionable weather hazards",
    "dangerous weather alert", "dangerous weather alerts",
)
KNOWLEDGE_MARKERS = (
    "typically", "usually", "what causes", "what does", "why does", "how does",
    "what is", "what are", "associated with", "meaning of", "mean", "means",
    "does ... mean", "wmo", "weather code", "weather codes", "uncertainty",
    "future weather claims",
)
STRONG_CONCEPTUAL_MARKERS = (
    "typically", "usually", "what causes", "what does", "why does", "how does",
    "associated with", "meaning of", "does ... mean", "wmo", "weather code",
    "weather codes", "uncertainty", "future weather claims",
)
RISK_MARKERS = ("safe", "risk", "suitable", "should i", "play", "run", "hike", "travel", "outdoor")
CURRENT_MARKERS = ("current weather", "current conditions", "weather right now", "weather now")


def _contains_marker(text: str, marker: str) -> bool:
    """Match phrases as words, avoiding substring false positives (e.g. play/display)."""
    marker = marker.strip().lower()
    if not marker:
        return False
    if marker.startswith(" vs ") and marker.endswith(" "):
        marker = marker.strip()
    escaped = re.escape(marker).replace(r"\ ", r"\s+")
    return bool(re.search(rf"(?<!\w){escaped}(?!\w)", text))


def _any_marker(text: str, markers: tuple[str, ...]) -> bool:
    return any(_contains_marker(text, marker) for marker in markers)


def classify(query: str) -> str:
    text = query.lower().strip()
    if _any_marker(text, COMPARISON_MARKERS):
        return "comparison"
    if _any_marker(text, RISK_MARKERS):
        return "activity_risk"
    if _any_marker(text, ALERT_MARKERS):
        return "alerts"
    # Strong conceptual markers override live words such as "forecast".
    # Generic "what is/what are" does not: "What is the forecast for Delhi
    # tomorrow?" must remain a live-weather request.
    if _any_marker(text, STRONG_CONCEPTUAL_MARKERS):
        return "knowledge"
    # Current-weather requests are a distinct deterministic capability. This
    # check must happen before the generic LIVE_MARKERS check so "right now"
    # cannot fall through to the broader live-weather route.
    if _any_marker(text, CURRENT_MARKERS):
        return "current_weather"
    if _any_marker(text, LIVE_MARKERS):
        return "live_weather"
    if _any_marker(text, KNOWLEDGE_MARKERS):
        return "knowledge"
    return "weather_intelligence"


def is_simple_current(query: str) -> bool:
    """Return whether the request is explicitly for present conditions.

    Do not reject "right now" merely because it is also a live-time marker;
    it is one of the canonical current-weather phrasings.
    """
    return _any_marker(query.lower().strip(), CURRENT_MARKERS)
