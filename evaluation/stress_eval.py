"""Balanced 100-case evaluation suite for the real Gemini + MCP agent.

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
    "What is the current weather in {city}?",
    "How is the weather right now in {city}?",
    "Give me the current weather for {city}.",
    "What are conditions like currently in {city}?",
    "Tell me today's current weather in {city}.",
    "What is {city}'s weather at the moment?",
    "Show current conditions for {city}.",
    "I need the live weather for {city}.",
)
FORECAST = (
    "What will the weather be like in {city} tomorrow?",
    "Give me tomorrow's forecast for {city}.",
    "What is the forecast for {city} tomorrow?",
    "How will weather look tomorrow in {city}?",
    "Tell me the weather forecast for tomorrow in {city}.",
    "What conditions are expected in {city} tomorrow?",
    "Show tomorrow weather for {city}.",
    "Can you check tomorrow's weather in {city}?",
)
ALERTS = (
    "Are there any weather hazards or alerts for {city}?",
    "What weather risks should I watch for in {city}?",
    "Check weather alerts for {city}.",
    "Are there dangerous weather conditions in {city}?",
    "What hazards are forecast for {city}?",
    "Give me the weather warnings for {city}.",
    "Is there any severe weather expected in {city}?",
    "Check forecast hazards in {city}.",
)
RISK = (
    "Is outdoor activity safe in {city} tomorrow?",
    "Would running be risky in {city} tomorrow?",
    "Assess outdoor activity risk in {city} for tomorrow.",
    "Is it suitable for outdoor activity in {city} tomorrow?",
    "How risky is outdoor activity in {city} tomorrow?",
    "Should I avoid outdoor activity in {city} tomorrow?",
    "Assess the weather risk for outdoor activity in {city} tomorrow.",
    "Would outdoor exercise be safe in {city} tomorrow?",
)
KNOWLEDGE = (
    "What conditions are associated with heavy rainfall?",
    "What are the common hazards of thunderstorms?",
    "What precautions are useful during extreme heat?",
    "What weather conditions can reduce visibility?",
    "What are typical risks from strong winds?",
    "How can heavy rain affect outdoor activities?",
    "What precautions should people take during thunderstorms?",
    "What does high precipitation probability mean?",
)


def build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    families = (
        ("current", CURRENT, "get_weather", lambda city: {"location": city}),
        ("forecast", FORECAST, "get_forecast", lambda city: {"location": city, "date": "tomorrow"}),
        ("alerts", ALERTS, "get_weather_alerts", lambda city: {"location": city}),
        ("risk", RISK, "assess_weather_risk", lambda city: {"location": city, "date": "tomorrow"}),
    )

    # 20 cases per live intent family: two paraphrases for every city (16),
    # plus four additional paraphrases across the first four cities (4).
    for family, templates, tool, args_builder in families:
        selected = [(city, index) for index in range(2) for city in CITIES]
        selected.extend((CITIES[index], 2) for index in range(4))
        for index, (city, template_index) in enumerate(selected):
            template = templates[template_index]
            cases.append({
                "id": f"stress-{family}-{index:02d}",
                "category": family,
                "question": template.format(city=city),
                "expected_tools": [tool],
                "required_args": [{"tool": tool, **args_builder(city)}],
            })

    # 20 deterministic knowledge/RAG cases, cycling through the knowledge
    # prompts rather than issuing hundreds of near-duplicate live requests.
    for index in range(20):
        template = KNOWLEDGE[index % len(KNOWLEDGE)]
        cases.append({
            "id": f"stress-knowledge-{index:02d}",
            "category": "knowledge",
            "question": template,
            "expected_tools": ["search_weather", "ask_weather"],
            "required_args": [],
        })
    return cases


def _arg_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(str(actual.get(k, "")).strip().lower() == str(v).strip().lower() for k, v in expected.items() if k != "tool")


def _failure_detail(result: dict[str, Any]) -> str | None:
    """Expose useful structured failure context without leaking raw tool output."""
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
    """Flatten Python 3.11+ exception groups to their deepest useful cause."""
    if isinstance(exc, BaseExceptionGroup):
        details = [_exception_detail(child) for child in exc.exceptions]
        details = [item for item in details if item]
        return details[0] if details else "ExceptionGroup: no underlying exception detail"
    return f"{type(exc).__name__}: {exc}"


async def main() -> None:
    cases = build_cases()
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
        expected = case["expected_tools"]
        tool_ok = any(call.get("name") in expected for call in calls)
        required_args = case.get("required_args", [])
        args_ok = all(any(call.get("name") == item["tool"] and _arg_match(call.get("arguments") or {}, item) for call in calls) for item in required_args)
        rows.append({"id": case["id"], "category": case["category"], "success": bool(result.get("success")), "tool_selection_correct": tool_ok, "argument_accuracy": args_ok, "argument_evaluable": bool(required_args), "latency_ms": round(latency, 2), "error": error})
        if index % 5 == 0 or index == len(cases):
            print(f"[{index}/{len(cases)}] evaluated", flush=True)

    latencies = [row["latency_ms"] for row in rows]
    argument_rows = [row for row in rows if row["argument_evaluable"]]
    summary = {
        "cases": len(rows),
        "task_success_rate": round(sum(r["success"] for r in rows) / len(rows), 4),
        "tool_selection_accuracy": round(sum(r["tool_selection_correct"] for r in rows) / len(rows), 4),
        "argument_accuracy": round(sum(r["argument_accuracy"] for r in argument_rows) / len(argument_rows), 4) if argument_rows else None,
        "mean_latency_ms": round(statistics.mean(latencies), 2),
        "p50_latency_ms": round(statistics.median(latencies), 2),
        "p95_latency_ms": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 2),
    }
    report = {"suite": "balanced 100-case stress evaluation", "summary": summary, "rows": rows}
    path = os.environ.get("WEATHER_STRESS_REPORT", "evaluation/stress_report.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
