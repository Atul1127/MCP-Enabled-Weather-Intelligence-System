"""Evaluate MCP tool selection and argument extraction for the weather agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Allow `python evaluation/evaluate_agent.py` to import modules from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ollama import AsyncClient

from mcp_client import connect, discover_tools
from weather_agent import SYSTEM_PROMPT

MODEL = os.environ.get("WEATHER_AGENT_MODEL", "llama3.2:3b")
DATASET = Path(__file__).with_name("dataset.json")

ALLOWED_TOOLS = {
    "get_weather",
    "assess_weather_risk",
    "get_weather_alerts",
    "search_weather",
}


def location_match(expected: dict[str, Any], calls: list[dict[str, Any]]) -> bool:
    locations = expected.get("expected_locations")
    if locations is None:
        location = expected.get("expected_location")
        locations = [location] if location else []
    if not locations:
        return True
    actual = []
    for call in calls:
        loc = call.get("arguments", {}).get("location")
        if loc:
            actual.append(str(loc).lower())
    return all(any(str(want).lower() in got for got in actual) for want in locations)


async def evaluate_case(case: dict[str, Any], ollama: AsyncClient, tools: list[dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    response = await ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Select the MCP tool(s) for this request. Do not answer the request. "
                    "Return tool calls only.\n\nUser request: " + case["query"]
                ),
            },
        ],
        tools=tools,
        stream=False,
        options={"temperature": 0},
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    calls = []
    for tool_call in response.message.tool_calls or []:
        calls.append({
            "name": tool_call.function.name,
            "arguments": dict(tool_call.function.arguments or {}),
        })

    expected = set(case["expected_tools"])
    actual = {call["name"] for call in calls}

    return {
        "id": case["id"],
        "category": case.get("category", "uncategorized"),
        "query": case["query"],
        "expected_tools": sorted(expected),
        "actual_tools": sorted(actual),
        "tool_selection_exact": actual == expected,
        "expected_tools_found": expected.issubset(actual),
        "location_arguments_correct": location_match(case, calls),
        "latency_ms": latency_ms,
        "tool_calls": calls,
    }


def percentile(values: list[float], percentile_rank: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = (len(values) - 1) * percentile_rank
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    fraction = index - lower
    return round(values[lower] + (values[upper] - values[lower]) * fraction, 2)


def category_metrics(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["category"], []).append(result)

    metrics: dict[str, dict[str, Any]] = {}
    for category, cases in grouped.items():
        metrics[category] = {
            "count": len(cases),
            "exact_accuracy": round(
                sum(case["tool_selection_exact"] for case in cases) / len(cases), 4
            ),
            "tool_recall": round(
                sum(case["expected_tools_found"] for case in cases) / len(cases), 4
            ),
            "location_accuracy": round(
                sum(case["location_arguments_correct"] for case in cases) / len(cases), 4
            ),
        }
    return metrics


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate weather MCP agent tool selection.")
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--output", default="evaluation/results.json")
    args = parser.parse_args()

    cases = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    ollama = AsyncClient()

    async with connect() as session:
        discovered = await discover_tools(session)
        tools = [tool for tool in discovered if tool["function"]["name"] in ALLOWED_TOOLS]
        results = [await evaluate_case(case, ollama, tools) for case in cases]

    exact = sum(r["tool_selection_exact"] for r in results)
    expected_hit = sum(r["expected_tools_found"] for r in results)
    location_ok = sum(r["location_arguments_correct"] for r in results)
    latencies = [r["latency_ms"] for r in results]

    report = {
        "model": MODEL,
        "dataset_size": len(results),
        "metrics": {
            "tool_selection_exact_accuracy": round(exact / len(results), 4),
            "expected_tool_recall": round(expected_hit / len(results), 4),
            "location_argument_accuracy": round(location_ok / len(results), 4),
            "average_tool_selection_latency_ms": round(sum(latencies) / len(latencies), 2),
            "p50_tool_selection_latency_ms": percentile(latencies, 0.50),
            "p95_tool_selection_latency_ms": percentile(latencies, 0.95),
        },
        "category_metrics": category_metrics(results),
        "cases": results,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Weather Agent Evaluation")
    print("=" * 28)
    for key, value in report["metrics"].items():
        print(f"{key}: {value}")
    print("\nCategory Metrics")
    print("-" * 28)
    for category, values in report["category_metrics"].items():
        print(f"{category}: {values['exact_accuracy']:.2%} ({values['count']} cases)")
    print(f"\nResults: {output}")


if __name__ == "__main__":
    asyncio.run(main())
