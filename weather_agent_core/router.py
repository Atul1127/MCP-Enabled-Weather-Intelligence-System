"""Deterministic first-pass intent router."""
from __future__ import annotations

import re


LIVE_MARKERS = (
    "today", "tomorrow", "now", "right now", "tonight", "this evening",
    "forecast", "next week", "next few days",
)
COMPARISON_MARKERS = ("compare", "versus", " vs ", "between ", "which is better")
ALERT_MARKERS = (
    "weather alert", "weather alerts", "weather hazard", "weather hazards",
    "dangerous weather", "actionable weather hazard", "actionable weather hazards",
    "dangerous weather alert", "dangerous weather alerts",
    "what weather risks should i watch for", "what weather risk should i watch for",
    "risks should i watch for", "risk should i watch for", "weather warnings",
    "weather warning", "warnings for", "warning for", "severe weather expected",
)
KNOWLEDGE_MARKERS = (
    "typically", "usually", "what causes", "what does", "why does", "how does",
    "what is", "what are", "associated with", "meaning of", " mean", "means",
    "does ... mean", "wmo", "weather code", "weather codes", "uncertainty",
    "future weather claims", "precautions", "precaution", "what to do", "how to prepare",
    "prepare for", "prepared for", "safety measures", "safety tips", "protect yourself",
    "stay safe", "hazards of", "hazard of", "risks of", "risk of", "typical risks",
    "common risks", "effects of", "impact of", "during extreme", "during heavy",
    "during severe", "during thunderstorms", "in extreme heat", "heatwave", "heat wave",
    "reduce visibility", "visibility", "affect outdoor activities", "weather conditions can",
)
STRONG_CONCEPTUAL_MARKERS = (
    "typically", "usually", "what causes", "what does", "why does", "how does",
    "associated with", "meaning of", "does ... mean", "wmo", "weather code", "weather codes",
    "uncertainty", "future weather claims", "precautions", "precaution", "what to do",
    "how to prepare", "prepare for", "prepared for", "safety measures", "safety tips",
    "protect yourself", "stay safe", "hazards of", "hazard of", "risks of", "risk of",
    "typical risks", "common risks", "effects of", "impact of", "during extreme",
    "during heavy", "during severe", "during thunderstorms", "in extreme heat", "heatwave",
    "heat wave", "reduce visibility", "visibility", "affect outdoor activities", "weather conditions can",
)
RISK_MARKERS = (
    "safe", "risk", "risky", "suitable", "should i", "play", "run", "running", "hike", "travel", "outdoor",
)
CURRENT_MARKERS = (
    "current weather", "current conditions", "weather right now", "weather now",
    "currently", "at the moment", "at present", "right at this moment",
)


def _contains_marker(text: str, marker: str) -> bool:
    marker = marker.strip().lower()
    if not marker:
        return False
    escaped = re.escape(marker).replace(r"\ ", r"\s+")
    return bool(re.search(rf"(?<!\w){escaped}(?!\w)", text))


def _any_marker(text: str, markers: tuple[str, ...]) -> bool:
    return any(_contains_marker(text, marker) for marker in markers)


def classify(query: str) -> str:
    text = query.lower().strip()
    if _any_marker(text, COMPARISON_MARKERS):
        return "comparison"
    # Explicit alert/warning questions must win over generic "risk" wording.
    if _any_marker(text, ALERT_MARKERS):
        return "alerts"
    # Explicit present-condition phrasing wins over broad conceptual words.
    if _any_marker(text, CURRENT_MARKERS):
        return "current_weather"
    # Conceptual questions are checked before activity-risk markers so phrases
    # like "affect outdoor activities" and "precautions during thunderstorms"
    # remain knowledge/RAG requests.
    if _any_marker(text, STRONG_CONCEPTUAL_MARKERS):
        return "knowledge"
    if _any_marker(text, RISK_MARKERS):
        return "activity_risk"
    if _any_marker(text, LIVE_MARKERS):
        return "live_weather"
    if _any_marker(text, KNOWLEDGE_MARKERS):
        return "knowledge"
    return "weather_intelligence"


def is_simple_current(query: str) -> bool:
    return _any_marker(query.lower().strip(), CURRENT_MARKERS)
