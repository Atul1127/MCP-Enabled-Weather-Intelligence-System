"""Reproducible end-to-end benchmark for the complete WeatherAgent."""
from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from weather_agent_core import WeatherAgent

from .exception_utils import classify_exception

DATASET = Path(__file__).with_name("agent_eval_dataset.json")
REPORT = Path(__file__).with_name("agent_e2e_report.json")


def _norm(value: Any) -> str:
    return " ".join(str(value or "").lower().strip().split())


def _argument_match(expected: dict[str, Any], actual_calls: list[dict[str, Any]]) -> bool:
    """Match required arguments while allowing harmless activity phrasing variants."""
    tool = expected.get("tool")
    for call in actual_calls:
        if call.get("name") != tool:
            continue
        args = call.get("arguments") or {}
        matched = True
        for key, value in expected.items():
            if key == "tool":
                continue
            actual = args.get(key)
            if key == "activity":
                expected_tokens = set(_norm(value).split())
                actual_tokens = set(_norm(actual).split())
                if not expected_tokens or len(actual_tokens & expected_tokens) / len(expected_tokens) < 0.5:
                    matched = False
                    break
            elif _norm(actual) != _norm(value):
                matched = False
                break
        if matched:
            return True
    return False


def _leaf_exceptions(exc: BaseException) -> list[BaseException]:
    """Unwrap ExceptionGroup/TaskGroup failures so root causes are visible."""
    nested = getattr(exc, "exceptions", None)
    if not nested:
        return [exc]
    leaves: list[BaseException] = []
    for child in nested:
        leaves.extend(_leaf_exceptions(child))
    return leaves


def _exception_details(exc: BaseException) -> tuple[str, str]:
    leaves = _leaf_exceptions(exc)
    primary = leaves[0] if leaves else exc
    detail = f"{type(primary).__name__}: {primary}"
    if len(leaves) > 1:
        detail += f" (+{len(leaves) - 1} nested exception(s))"
    return classify_exception(primary), detail


def evaluate_case(case: dict[str, Any], result: dict[str, Any], latency_ms: float) -> dict[str, Any]:
    successful = {
        str(item.get("tool"))
        for item in result.get("observations", [])
        if item.get("tool")
        and not (isinstance(item.get("result"), dict) and item["result"].get("success") is False)
    }
    expected_tools = set(case.get("expected_tools", []))
    tool_recall = len(successful & expected_tools) / len(expected_tools) if expected_tools else 1.0
    required_args = case.get("required_args", [])
    arg_accuracy = (
        sum(_argument_match(item, result.get("tool_calls", [])) for item in required_args) / len(required_args)
        if required_args else None
    )
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
            failure_type = None
            exception_message = None
        except Exception as exc:
            failure_type, exception_message = _exception_details(exc)
            result = {"success": False, "answer": "", "errors": [exception_message]}
        latency_ms = (time.perf_counter() - started) * 1000
        row = evaluate_case(case, result, latency_ms)
        row["exception"] = failure_type
        row["exception_message"] = exception_message
        row["infrastructure_failure"] = failure_type in {"quota_exhausted", "authentication_failure", "network_failure"}
        rows.append(row)

    latencies = sorted(row["latency_ms"] for row in rows)
    exception_counts = Counter(row["exception"] for row in rows if row["exception"])
    infrastructure_failures = sum(row["infrastructure_failure"] for row in rows)
    agent_failures = sum(not row["success"] and not row["infrastructure_failure"] for row in rows)
    first_failure = next(({"id": row["id"], "exception": row["exception_message"], "failure_type": row["exception"]} for row in rows if row["exception_message"]), None)

    category_summary: dict[str, dict[str, float | int]] = {}
    for category in sorted({str(row["category"]) for row in rows}):
        group = [row for row in rows if row["category"] == category]
        category_summary[category] = {
            "cases": len(group),
            "success_rate": round(sum(row["success"] for row in group) / len(group), 4),
            "tool_selection_recall": round(statistics.mean(row["tool_selection_recall"] for row in group), 4),
            "evidence_sufficiency_rate": round(sum(row["evidence_sufficient"] for row in group) / len(group), 4),
        }

    summary = {
        "cases": len(rows),
        "task_success_rate": round(sum(row["success"] for row in rows) / len(rows), 4) if rows else 0.0,
        "answer_rate": round(sum(row["answer_present"] for row in rows) / len(rows), 4) if rows else 0.0,
        "mean_tool_selection_recall": _mean(rows, "tool_selection_recall"),
        "mean_argument_accuracy": _mean(rows, "argument_accuracy"),
        "evidence_sufficiency_rate": round(sum(row["evidence_sufficient"] for row in rows) / len(rows), 4) if rows else 0.0,
        "error_rate": round(sum(row["error_count"] > 0 or row["exception"] is not None for row in rows) / len(rows), 4) if rows else 0.0,
        "infrastructure_failure_rate": round(infrastructure_failures / len(rows), 4) if rows else 0.0,
        "agent_failure_rate": round(agent_failures / len(rows), 4) if rows else 0.0,
        "exception_counts": dict(exception_counts),
        "first_failure": first_failure,
        "category_summary": category_summary,
        "mean_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p50_latency_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": round(latencies[max(0, int(len(latencies) * 0.95) - 1)], 2) if latencies else 0.0,
    }

    payload = {
        "dataset": DATASET.name,
        "evaluation": "live end-to-end WeatherAgent benchmark",
        "metric_note": "Tool recall measures whether expected capability families were successfully invoked; argument accuracy checks explicitly labeled required arguments, with activity phrasing normalized consistently with the legacy benchmark. Infrastructure failures are reported separately from agent failures.",
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
