"""Deterministic first-pass intent router.

Routing rules, in priority order:

1. Comparisons ("compare", "vs", "between Delhi and Mumbai").
2. Explicit alert/warning requests.
3. Explicit present-condition phrasing ("weather right now").
4. A question that names a place AND a time ("in Mumbai tomorrow") always
   needs live data, even when it also contains conceptual words such as
   "risk of", "visibility" or "heatwave". If it also asks for an explanation
   or advice, it is routed to ``live_knowledge`` (live data + knowledge base).
5. A question that names a place and asks about a measurable condition
   ("visibility in Delhi", "is it raining in Kolkata") is live weather.
6. Otherwise the original conceptual / activity-risk / live / knowledge order.
"""
from __future__ import annotations

import re


LIVE_MARKERS = (
    "today", "tomorrow", "now", "right now", "tonight", "this evening",
    "forecast", "next week", "next few days",
)
# Time references that make a place-specific question a live-data question.
TIME_MARKERS = (
    "today", "tomorrow", "tonight", "now", "right now", "currently",
    "this morning", "this afternoon", "this evening", "this week", "this weekend",
    "next week", "next few days", "coming days", "later today", "day after tomorrow",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
)
COMPARISON_MARKERS = ("compare", "versus", " vs ", "vs.", "which is better")
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
# Markers that ask for an explanation or advice. When these appear together
# with a place and a time, the question needs live data AND the knowledge base.
EXPLANATION_MARKERS = (
    "why", "what causes", "how does", "typically", "usually", "meaning of", " mean",
    "means", "precautions", "precaution", "what to do", "how to prepare", "prepare for",
    "safety measures", "safety tips", "protect yourself", "stay safe",
)
RISK_MARKERS = (
    "safe", "risk", "risky", "suitable", "should i", "play", "run", "running", "hike", "travel", "outdoor",
)
# "Risk/chance of rain" asks for a forecast probability, not an activity assessment.
CHANCE_PHRASE = re.compile(r"\b(?:risk|chance|chances|probability|likelihood|odds)\s+of\b")
CURRENT_MARKERS = (
    "current weather", "current conditions", "weather right now", "weather now",
    "currently", "at the moment", "at present", "right at this moment",
)
# Measurable conditions that, asked about a named place, mean live data.
MEASUREMENT_MARKERS = (
    "temperature", "humidity", "visibility", "wind", "windy", "rain", "raining",
    "rainfall", "hot", "cold", "warm", "sunny", "cloudy", "foggy", "fog",
    "storm", "stormy", "heatwave", "heat wave", "uv", "weather",
)

# Words that end a place name ("in Mumbai tomorrow", "in Pune between 4 and 6").
_PLACE_STOP_WORDS = {
    "today", "tomorrow", "tonight", "now", "this", "next", "later", "currently",
    "on", "at", "in", "for", "during", "between", "and", "or", "vs", "versus",
    "the", "morning", "afternoon", "evening", "night", "week", "weekend",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "right", "around", "near", "from", "to", "by", "with", "if", "when", "while",
    "please", "pm", "am",
}
_PREPOSITIONS = {"in", "for", "at", "near", "around", "of"}
_COMPARISON_BETWEEN = re.compile(r"\bbetween\s+([A-Za-z][\w.'-]*)(?:\s+[A-Za-z][\w.'-]*){0,2}\s+and\s+([A-Za-z])")
_NON_PLACE_HEADS = {
    "rain", "rainfall", "heavy", "extreme", "the", "a", "an", "my", "our", "your",
    "weather", "heat", "thunderstorms", "thunderstorm", "storms", "storm", "fog",
    "visibility", "humidity", "wind", "temperature", "outdoor", "outdoors", "general",
    "india", "monsoon", "summer", "winter", "spring", "autumn", "july", "june", "august",
    "precipitation", "case", "terms", "order", "advance", "front", "place", "time",
}
_LEADING_ARTICLES = {"the", "a", "an"}


def _contains_marker(text: str, marker: str) -> bool:
    marker = marker.strip().lower()
    if not marker:
        return False
    escaped = re.escape(marker).replace(r"\ ", r"\s+")
    return bool(re.search(rf"(?<!\w){escaped}(?!\w)", text))


def _any_marker(text: str, markers: tuple[str, ...]) -> bool:
    return any(_contains_marker(text, marker) for marker in markers)


def extract_location(query: str) -> str | None:
    """Return the most likely place name in ``query``, or None.

    Looks at the words after each preposition ("in", "for", "at", ...) and
    stops at the first time or connector word, so "in Mumbai in the evening"
    gives "Mumbai" rather than "the evening".
    """
    tokens = [word.strip(".'-") for word in re.split(r"[\s?!,;:]+", query) if word.strip(".'-")]
    candidates: list[str] = []
    for index, token in enumerate(tokens):
        if token.lower() not in _PREPOSITIONS:
            continue
        words: list[str] = []
        for word in tokens[index + 1:index + 5]:
            lowered = word.lower()
            if lowered in _PLACE_STOP_WORDS or word.isdigit() or not word[:1].isalpha():
                if not words and lowered in _LEADING_ARTICLES:
                    continue
                break
            words.append(word)
        if not words or words[0].lower() in _NON_PLACE_HEADS:
            continue
        candidates.append(" ".join(words))
    if not candidates:
        return None
    # Prefer a capitalised candidate (a proper noun); otherwise the first one.
    for candidate in candidates:
        if candidate[:1].isupper():
            return candidate
    return candidates[0]


def _has_place(query: str) -> bool:
    return extract_location(query) is not None


def _is_comparison(text: str) -> bool:
    if _any_marker(text, COMPARISON_MARKERS):
        return True
    # "between Delhi and Mumbai" is a comparison; "between 4 and 6 pm" and
    # "between morning and evening" are time ranges, not comparisons.
    match = _COMPARISON_BETWEEN.search(text)
    return bool(match) and match.group(1).lower() not in _PLACE_STOP_WORDS


def _activity_risk(text: str) -> bool:
    without_chance = CHANCE_PHRASE.sub(" ", text)
    return _any_marker(without_chance, RISK_MARKERS)


def classify(query: str) -> str:
    text = query.lower().strip()
    if _is_comparison(text):
        return "comparison"
    # Explicit alert/warning questions must win over generic "risk" wording.
    if _any_marker(text, ALERT_MARKERS):
        return "alerts"
    # Explicit present-condition phrasing wins over broad conceptual words.
    if _any_marker(text, CURRENT_MARKERS):
        return "current_weather"

    has_place = _has_place(query)
    has_time = _any_marker(text, TIME_MARKERS)
    explanation = _any_marker(text, EXPLANATION_MARKERS)

    # A named place plus a time always needs live data.
    if has_place and has_time:
        if explanation:
            return "live_knowledge"
        if _activity_risk(text):
            return "activity_risk"
        return "live_weather"
    # "What is the visibility in Delhi?" asks about current conditions.
    if has_place and _any_marker(text, MEASUREMENT_MARKERS) and not explanation:
        if _activity_risk(text):
            return "activity_risk"
        return "live_weather"

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
