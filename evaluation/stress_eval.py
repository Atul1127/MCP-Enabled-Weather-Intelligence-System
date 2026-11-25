"""Balanced 100-case evaluation suite plus 30 held-out questions for the real Gemini + MCP agent.

The suite uses deterministic paraphrases and multiple Indian cities so the
benchmark covers the four main live-weather intent families plus knowledge/RAG.
Run only when Gemini credentials and MCP dependencies are configured; these are
live provider calls and can consume quota.
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from typing import Any

from agent import run_agent

CITIES = ("Kolkata", "Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Pune", "Jaipur")
CURRENT = (
    "What is the current weather in {city}?", "How is the weather right now in {city}?",
    "Give me the current weather for {city}.", "What are conditions like currently in {city}?",
    "Tell me today's current weather in {city}.", "What is {city}'s weather at the moment?",
    "Show current conditions for {city}.", "I need the live weather for {city}.",
)
FORECAST = (
    "What will the weather be like in {city} tomorrow?", "Give me tomorrow's forecast for {city}.",
    "What is the forecast for {city} tomorrow?", "How will weather look tomorrow in {city}?",
    "Tell me the weather forecast for tomorrow in {city}.", "What conditions are expected in {city} tomorrow?",
    "Show tomorrow weather for {city}.", "Can you check tomorrow's weather in {city}?",
)
ALERTS = (
    "Are there any weather hazards or alerts for {city}?", "What weather risks should I watch for in {city}?",
    "Check weather alerts for {city}.", "Are there dangerous weather conditions in {city}?",
    "What hazards are forecast for {city}?", "Give me the weather warnings for {city}.",
    "Is there any severe weather expected in {city}?", "Check forecast hazards in {city}.",
)
RISK = (
    "Is outdoor activity safe in {city} tomorrow?", "Would running be risky in {city} tomorrow?",
    "Assess outdoor activity risk in {city} for tomorrow.", "Is it suitable for outdoor activity in {city} tomorrow?",
    "How risky is outdoor activity in {city} tomorrow?", "Should I avoid outdoor activity in {city} tomorrow?",
    "Assess the weather risk for outdoor activity in {city} tomorrow.", "Would outdoor exercise be safe in {city} tomorrow?",
)
KNOWLEDGE = (
    "What conditions are associated with heavy rainfall?", "What are the common hazards of thunderstorms?",
    "What precautions are useful during extreme heat?", "What weather conditions can reduce visibility?",
    "What are typical risks from strong winds?", "How can heavy rain affect outdoor activities?",
    "What precautions should people take during thunderstorms?", "What does high precipitation probability mean?",
)


# Held-out questions written in natural phrasing, NOT generated from the
# templates above. Several of these were misrouted by the original router, so
# they are the ones that show whether routing generalises.
HELD_OUT: tuple[tuple[str, str, list[str], dict[str, str] | None], ...] = (
    ("held-risk-of-rain", "What is the risk of rain in Mumbai tomorrow?", ["get_forecast", "assess_weather_risk"], {"location": "Mumbai"}),
    ("held-visibility", "What is the visibility in Delhi today?", ["get_weather", "get_forecast"], {"location": "Delhi"}),
    ("held-heatwave-week", "Is a heatwave expected in Delhi this week?", ["get_forecast", "get_weather_alerts"], {"location": "Delhi"}),
    ("held-rain-window", "Will it rain between 4 and 6 pm in Pune?", ["get_forecast", "get_weather"], {"location": "Pune"}),
    ("held-umbrella", "Do I need an umbrella in Kolkata tomorrow?", ["get_forecast", "assess_weather_risk"], {"location": "Kolkata"}),
    ("held-jacket", "Should I carry a jacket in Shimla tonight?", ["get_weather", "get_forecast"], {"location": "Shimla"}),
    ("held-humidity", "How humid is it in Chennai right now?", ["get_weather"], {"location": "Chennai"}),
    ("held-wind", "How windy will it be in Goa tomorrow?", ["get_forecast"], {"location": "Goa"}),
    ("held-weekend", "What's the weekend looking like in Bengaluru?", ["get_forecast"], {"location": "Bengaluru"}),
    ("held-cricket", "Can we play cricket in Hyderabad tomorrow evening?", ["assess_weather_risk", "get_forecast"], {"location": "Hyderabad"}),
    ("held-cycling", "Is it a good day for cycling in Pune?", ["assess_weather_risk", "get_forecast", "get_weather"], {"location": "Pune"}),
    ("held-flight", "Any chance of a storm delaying flights out of Delhi tomorrow?", ["get_forecast", "get_weather_alerts", "assess_weather_risk"], {"location": "Delhi"}),
    ("held-compare", "Which is hotter today, Jaipur or Ahmedabad?", ["get_weather", "get_forecast"], None),
    ("held-compare-rain", "Compare tomorrow's rain in Mumbai and Pune.", ["get_forecast"], None),
    ("held-precautions-live", "What precautions should I take in Mumbai tomorrow?", ["get_forecast", "get_weather_alerts", "search_weather"], {"location": "Mumbai"}),
    ("held-why-hot", "Why is it so hot in Delhi today?", ["get_weather", "get_forecast"], {"location": "Delhi"}),
    ("held-morning", "What will the temperature be in Lucknow tomorrow morning?", ["get_forecast"], {"location": "Lucknow"}),
    ("held-lowercase-city", "weather in nagpur now", ["get_weather"], {"location": "nagpur"}),
    ("held-typo", "whats the forcast for chennai tomorow", ["get_forecast"], {"location": "chennai"}),
    ("held-feels-like", "What does it feel like outside in Kochi?", ["get_weather"], {"location": "Kochi"}),
    ("held-uv-picnic", "Is tomorrow okay for a picnic in Mysuru?", ["assess_weather_risk", "get_forecast"], {"location": "Mysuru"}),
    ("held-flood", "Is there a flood risk in Guwahati this week?", ["get_forecast", "get_weather_alerts", "assess_weather_risk"], {"location": "Guwahati"}),
    ("held-trek", "Is it safe to trek near Manali tomorrow?", ["assess_weather_risk", "get_forecast"], {"location": "Manali"}),
    ("held-knowledge-cyclone", "How do cyclones form over the Bay of Bengal?", ["search_weather", "ask_weather"], None),
    ("held-knowledge-heat-index", "What is the difference between temperature and feels-like temperature?", ["search_weather", "ask_weather"], None),
    ("held-knowledge-orange", "What does an orange alert mean?", ["search_weather", "ask_weather"], None),
    ("held-knowledge-monsoon", "Why does the monsoon bring heavy rain to the Western Ghats?", ["search_weather", "ask_weather"], None),
    ("held-knowledge-fog", "What causes dense fog in north India in winter?", ["search_weather", "ask_weather"], None),
    ("held-no-location", "Will it rain tomorrow?", [], None),
    ("held-off-topic", "Write me a poem about the stock market.", [], None),
)


def build_held_out_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for case_id, question, tools, args in HELD_OUT:
        required = []
        if args and tools:
            # Location must appear on at least one of the acceptable tools.
            required = [{"any_tool": tools, **args}]
        cases.append({
            "id": case_id, "category": "held_out", "held_out": True,
            "question": question, "expected_tools": tools,
            # An empty expected list means: the agent should NOT call a live tool
            # (ask for a location, or refuse off-topic) — success is judged on that.
            "expect_no_tools": not tools,
            "required_args": required,
        })
    return cases


def build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    families = (
        ("current", CURRENT, "get_weather", lambda city: {"location": city}),
        ("forecast", FORECAST, "get_forecast", lambda city: {"location": city, "date": "tomorrow"}),
        ("alerts", ALERTS, "get_weather_alerts", lambda city: {"location": city}),
        ("risk", RISK, "assess_weather_risk", lambda city: {"location": city, "date": "tomorrow"}),
    )
    for family, templates, tool, args_builder in families:
        selected = [(city, index) for index in range(2) for city in CITIES]
        selected.extend((CITIES[index], 2) for index in range(4))
        for index, (city, template_index) in enumerate(selected):
            cases.append({
                "id": f"stress-{family}-{index:02d}", "category": family,
                "question": templates[template_index].format(city=city),
                "expected_tools": [tool],
                "required_args": [{"tool": tool, **args_builder(city)}],
            })
    for index in range(20):
        cases.append({
            "id": f"stress-knowledge-{index:02d}", "category": "knowledge",
            "question": KNOWLEDGE[index % len(KNOWLEDGE)],
            "expected_tools": ["search_weather", "ask_weather"], "required_args": [],
        })
    return cases


def _arg_match(actual: dict[str, Any], expected: dict[str, Any], loose: bool = False) -> bool:
    for key, value in expected.items():
        if key in ("tool", "any_tool"):
            continue
        got = str(actual.get(key, "")).strip().lower()
        want = str(value).strip().lower()
        # Held-out cases accept "Mumbai, Maharashtra" for "Mumbai".
        if (want not in got) if loose else (got != want):
            return False
    return True


def _args_ok(calls: list[dict[str, Any]], required: list[dict[str, Any]]) -> bool:
    for item in required:
        names = item.get("any_tool") or [item["tool"]]
        loose = "any_tool" in item
        if not any(call.get("name") in names and _arg_match(call.get("arguments") or {}, item, loose) for call in calls):
            return False
    return True


def _model_calls(calls: list[dict[str, Any]], added: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool calls Gemini chose itself: all calls minus the ones code appended."""
    remaining = list(added)
    chosen = []
    for call in calls:
        match = next((a for a in remaining if a.get("name") == call.get("name")
                      and (a.get("arguments") or {}) == (call.get("arguments") or {})), None)
        if match is not None:
            remaining.remove(match)
        else:
            chosen.append(call)
    return chosen


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    return round(sum(bool(r[key]) for r in rows) / len(rows), 4) if rows else None


def _is_quota_failure(detail: str | None) -> bool:
    text = (detail or "").lower()
    return any(marker in text for marker in (
        "quota exhausted", "resource exhausted", "quota_exceeded", "resource_exhausted",
        "429", "rate limit", "too many requests",
    ))


def _failure_detail(result: dict[str, Any]) -> str | None:
    if result.get("success"):
        return None
    errors = result.get("errors") or []
    if errors:
        return "; ".join(str(item) for item in errors[:3])
    verification = result.get("verification") or {}
    if verification and not verification.get("sufficient", True):
        missing = verification.get("missing_capabilities") or []
        if missing:
            return "missing required evidence: " + ", ".join("/".join(map(str, group)) for group in missing)
    return "agent returned success=false"


def _exception_detail(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        details = [_exception_detail(child) for child in exc.exceptions]
        return details[0] if details else "ExceptionGroup: no underlying exception detail"
    return f"{type(exc).__name__}: {exc}"


async def main() -> None:
    cases = build_cases()
    if os.environ.get("WEATHER_STRESS_HELD_OUT", "1") != "0":
        cases += build_held_out_cases()
    limit = int(os.environ.get("WEATHER_STRESS_LIMIT", str(len(cases))))
    cases = cases[: max(1, min(limit, len(cases)))]
    rows: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        started = time.perf_counter()
        try:
            result = await run_agent(case["question"])
            error = _failure_detail(result)
        except Exception as exc:
            result = {"success": False, "tool_calls": []}
            error = _exception_detail(exc)
        latency = (time.perf_counter() - started) * 1000
        calls = result.get("tool_calls") or []
        added = result.get("code_added_tool_calls") or []
        chosen = _model_calls(calls, added)
        required_args = case.get("required_args", [])
        expected = case["expected_tools"]
        if case.get("expect_no_tools"):
            live = {"get_weather", "get_forecast", "get_weather_alerts", "assess_weather_risk"}
            tool_ok = not any(call.get("name") in live for call in calls)
            model_tool_ok = tool_ok
            success = tool_ok  # no live tool called = asked for a location / declined
        else:
            tool_ok = any(call.get("name") in expected for call in calls)
            model_tool_ok = any(call.get("name") in expected for call in chosen)
            success = bool(result.get("success"))
        quota_failure = _is_quota_failure(error)
        unverified = result.get("unverified_numbers") or []
        rows.append({
            "id": case["id"], "category": case["category"],
            "held_out": bool(case.get("held_out")),
            "question": case["question"],
            "intent": result.get("intent"),
            "success": success,
            "tool_selection_correct": tool_ok,
            "model_tool_selection_correct": model_tool_ok,
            "rescued_by_code": tool_ok and not model_tool_ok,
            "code_added_calls": [c.get("name") for c in added],
            "argument_accuracy": _args_ok(calls, required_args),
            "argument_evaluable": bool(required_args),
            "confidence": result.get("confidence"),
            "unverified_numbers": unverified,
            "has_unverified_numbers": bool(unverified),
            "provider_quota_failure": quota_failure,
            "latency_ms": round(latency, 2), "error": error,
        })
        if index % 5 == 0 or index == len(cases):
            print(f"[{index}/{len(cases)}] evaluated", flush=True)

    quality_rows = [row for row in rows if not row["provider_quota_failure"]]
    latencies = [row["latency_ms"] for row in quality_rows]
    if not latencies:
        latencies = [row["latency_ms"] for row in rows]
    argument_rows = [row for row in quality_rows if row["argument_evaluable"]]

    templated = [r for r in quality_rows if not r["held_out"]]
    held = [r for r in quality_rows if r["held_out"]]
    answered = [r for r in quality_rows if r["confidence"] is not None]
    summary = {
        "cases": len(rows),
        "templated_cases": sum(not r["held_out"] for r in rows),
        "held_out_cases": sum(r["held_out"] for r in rows),
        "provider_quota_failures": sum(row["provider_quota_failure"] for row in rows),
        "evaluated_without_provider_quota": len(quality_rows),
        "task_success_rate": _rate(quality_rows, "success"),
        "templated_task_success_rate": _rate(templated, "success"),
        "held_out_task_success_rate": _rate(held, "success"),
        # Any correct tool in the trace, including ones code appended.
        "tool_selection_accuracy": _rate(quality_rows, "tool_selection_correct"),
        # Only tools Gemini chose itself — the honest "model picked the tool" number.
        "model_tool_selection_accuracy": _rate(quality_rows, "model_tool_selection_correct"),
        "held_out_model_tool_selection_accuracy": _rate(held, "model_tool_selection_correct"),
        "rescued_by_code_rate": _rate(quality_rows, "rescued_by_code"),
        "argument_accuracy": _rate(argument_rows, "argument_accuracy"),
        "confidence_levels": {level: sum(r["confidence"] == level for r in answered)
                              for level in sorted({r["confidence"] for r in answered})},
        "answers_with_unverified_numbers_rate": _rate(answered, "has_unverified_numbers"),
        "mean_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
        "p50_latency_ms": round(statistics.median(latencies), 2) if latencies else None,
        "p95_latency_ms": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 2) if latencies else None,
        "total_task_success_rate_including_quota_failures": _rate(rows, "success"),
    }
    report = {"suite": "balanced 100-case stress evaluation + 30 held-out natural questions", "summary": summary, "rows": rows}
    path = os.environ.get("WEATHER_STRESS_REPORT", "evaluation/stress_report.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
