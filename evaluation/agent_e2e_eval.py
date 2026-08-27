"""Reproducible end-to-end benchmark for the complete WeatherAgent."""
from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

from weather_agent_core import WeatherAgent

DATASET = Path(__file__).with_name("agent_eval_dataset.json")
REPORT = Path(__file__).with_name("agent_e2e_report.json")


def _successful_tools(result: dict[str, Any]) -> set[str]:
    return {
        str(item.get("tool"))
        for item in result.get("observations", [])
        if item.get("tool")
        and not (isinstance(item.get("result"), dict) and item["result"].get("success") is False)
    }


def _argument_match(expected: dict[str, Any], actual_calls: list[dict[str, Any]]) -> bool:
    tool = expected.get("tool")
    for call in actual_calls:
        if call.get("name") != tool:
            continue
        args = call.get("arguments") or {}
        if all(str(args.get(key, "")).strip().lower() == str(value).strip().lower() for key, value in expected.items() if key != "tool"):
            return True
    return False


def evaluate_case(case: dict[str, Any], result: dict[str, Any], latency_ms: float) -> dict[str, Any]:
    successful = _successful_tools(result)
    expected_tools = set(case.get("expected_tools", []))
    tool_recall = len(successful & expected_tools) / len(expected_tools) if expected_tools else 1.0
    required_args = case.get("required_args", [])
    arg_accuracy = sum(_argument_match(item, result.get("tool_calls", [])) for item in required_args) / len(required_args) if required_args else None
    verification = result.get("verification") or {}
    citations = result.get("citations") or []
    answer = str(result.get("answer") or "").strip()
    return {
        "id": case["id"],
        "category": case.get("category", "unknown"),
        "success": bool(result.get("success")) and bool(answer),
        "answer_present": bool(answer),
        "tool_selection_recall": round(tool_recall, 4),
        "argument_accuracy": round(arg_accuracy, 4) if arg_accuracy is not None else None,
        "evidence_sufficient": bool(verification.get("sufficient", False)),
        "citation_count": len(citations),
        "error_count": len(result.get("errors") or []),
        "rounds": int(result.get("rounds", 0)),
        "retries": int(result.get("retry_count", 0)),
        "latency_ms": round(latency_ms, 2),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return round(statistics.mean(values), 4) if values else None


async def _run() -> dict[str, Any]:
    cases = json.loads(DATASET.read_text(encoding="utf-8"))["cases"]
    agent = WeatherAgent()
    rows: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        try:
            result = await agent.run(case["question"])
            error = None
        except Exception as exc:  # benchmark must record failures without stopping the suite
            result = {"success": False, "answer": "", "errors": [type(exc).__name__ + ": " + str(exc)]}
            error = type(exc).__name__
        latency_ms = (time.perf_counter() - started) * 1000
        row = evaluate_case(case, result, latency_ms)
        row["exception"] = error
        rows.append(row)

    latencies = sorted(row["latency_ms"] for row in rows)
    summary = {
        "cases": len(rows),
        "task_success_rate": round(sum(row["success"] for row in rows) / len(rows), 4) if rows else 0.0,
        "answer_rate": round(sum(row["answer_present"] for row in rows) / len(rows), 4) if rows else 0.0,
        "mean_tool_selection_recall": _mean(rows, "tool_selection_recall"),
        "mean_argument_accuracy": _mean(rows, "argument_accuracy"),
        "evidence_sufficiency_rate": round(sum(row["evidence_sufficient"] for row in rows) / len(rows), 4) if rows else 0.0,
        "error_rate": round(sum(row["error_count"] > 0 or row["exception"] is not None for row in rows) / len(rows), 4) if rows else 0.0,
        "mean_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p50_latency_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": round(latencies[max(0, int(len(latencies) * 0.95) - 1)], 2) if latencies else 0.0,
    }
    payload = {
        "dataset": DATASET.name,
        "evaluation": "live end-to-end WeatherAgent benchmark",
        "metric_note": "Tool recall measures whether expected capability families were successfully invoked; argument accuracy checks only explicitly labeled required arguments.",
        "summary": summary,
        "rows": rows,
    }
    REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return payload


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
