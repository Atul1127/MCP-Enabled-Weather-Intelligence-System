"""Small, quota-friendly subset runner for the live WeatherAgent benchmark."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from weather_agent_core import WeatherAgent
from .agent_e2e_eval import DATASET, evaluate_case
from .exception_utils import classify_exception


async def _run(limit: int, category: str | None) -> dict[str, Any]:
    cases = json.loads(DATASET.read_text(encoding="utf-8"))["cases"]
    if category:
        cases = [case for case in cases if str(case.get("category")) == category]
    cases = cases[:limit]
    if not cases:
        raise SystemExit("No evaluation cases matched the requested filter")

    agent = WeatherAgent()
    rows: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        try:
            result = await agent.run(case["question"])
            failure_type = None
            exception_message = None
        except Exception as exc:
            failure_type = classify_exception(exc)
            exception_message = f"{type(exc).__name__}: {exc}"
            result = {"success": False, "answer": "", "errors": [exception_message]}
        latency_ms = (time.perf_counter() - started) * 1000
        row = evaluate_case(case, result, latency_ms)
        row["exception"] = failure_type
        row["exception_message"] = exception_message
        rows.append(row)

    summary = {
        "cases": len(rows),
        "task_success_rate": round(sum(row["success"] for row in rows) / len(rows), 4),
        "mean_tool_selection_recall": round(sum(row["tool_selection_recall"] for row in rows) / len(rows), 4),
        "evidence_sufficiency_rate": round(sum(row["evidence_sufficient"] for row in rows) / len(rows), 4),
        "rows": rows,
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small live WeatherAgent evaluation subset")
    parser.add_argument("--limit", type=int, default=1, help="Maximum number of cases to execute")
    parser.add_argument("--category", default=None, help="Optional exact dataset category filter")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    asyncio.run(_run(args.limit, args.category))


if __name__ == "__main__":
    main()
