"""Deterministic checks on the final answer.

* ``evidence_status`` replaces the model's self-reported confidence with a
  level the code can justify: "complete", "partial" or "no_live_data".
* ``unverified_numbers`` finds weather numbers in the answer (°C, %, mm, km/h)
  that do not appear anywhere in the evidence the tools returned.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

LIVE_TOOLS = {"get_weather", "get_forecast", "get_weather_alerts", "assess_weather_risk"}
SEARCH_TOOLS = {"search_weather", "ask_weather"}

_NUMBER_WITH_UNIT = re.compile(
    r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*(°\s*C|℃|degrees?\s*(?:celsius|c)?|%|percent|mm|km\s*/\s*h|kmph|kph)",
    re.I,
)
_ANY_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_CITATION = re.compile(r"\[S\d+\]")


def _succeeded(observation: dict[str, Any]) -> bool:
    result = observation.get("result")
    return not (isinstance(result, dict) and result.get("success") is False)


def evidence_status(plan: dict[str, Any], observations: list[dict[str, Any]], verification: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return {"level": ..., "reasons": [...]} from facts the code already knows."""
    verification = verification or {}
    reasons: list[str] = []

    live_ok = any(obs.get("tool") in LIVE_TOOLS and _succeeded(obs) for obs in observations)
    if plan.get("requires_live_data") and not live_ok:
        return {"level": "no_live_data", "reasons": ["No live weather data could be retrieved for this question."]}

    failed = sorted({str(obs.get("tool")) for obs in observations if not _succeeded(obs)})
    if failed:
        reasons.append("Some tools failed: " + ", ".join(failed) + ".")

    missing = verification.get("missing_capabilities") or []
    if verification and not verification.get("sufficient", True):
        groups = ", ".join("/".join(group) for group in missing) or "required evidence"
        reasons.append(f"Missing required evidence: {groups}.")

    searches = [obs for obs in observations if obs.get("tool") in SEARCH_TOOLS and _succeeded(obs)]
    if plan.get("requires_knowledge") and searches and not any((obs.get("result") or {}).get("documents") for obs in searches):
        reasons.append("The knowledge search returned no documents.")

    return {"level": "partial" if reasons else "complete", "reasons": reasons}


def _numbers_in(value: Any) -> Iterable[float]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, str):
        for match in _ANY_NUMBER.findall(value):
            try:
                yield float(match)
            except ValueError:
                continue
    elif isinstance(value, dict):
        for item in value.values():
            yield from _numbers_in(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _numbers_in(item)


def unverified_numbers(answer: str, evidence: Any) -> list[str]:
    """Return weather values stated in ``answer`` that the evidence does not contain.

    A value counts as supported if some evidence number is within rounding
    distance of it (0.5, or 1% for large values), so "31°C" matches 30.8.
    """
    known = list(_numbers_in(evidence))
    text = _CITATION.sub(" ", answer or "")
    unverified: list[str] = []
    for match in _NUMBER_WITH_UNIT.finditer(text):
        value = float(match.group(1))
        tolerance = max(0.51, abs(value) * 0.01)
        if not any(abs(value - candidate) <= tolerance for candidate in known):
            phrase = match.group(0).strip()
            if phrase not in unverified:
                unverified.append(phrase)
    return unverified
