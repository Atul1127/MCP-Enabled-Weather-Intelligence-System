"""Evaluate the actual Gemini MCP weather agent's tool selection and arguments.

Run from the repository root:
    python evaluation/agent_benchmark.py

This is an end-to-end benchmark: each case calls run_agent(), so the metrics
measure the real Gemini + MCP tool-selection loop. Knowledge Graph is excluded.
"""
from __future__ import annotations
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from agent import run_agent
DATASET = Path(__file__).resolve().parent / "agent_eval_dataset.json"
REPORT = Path(__file__).resolve().parent / "agent_benchmark_report.json"

def norm(value: Any) -> str:
    return " ".join(str(value or "").lower().strip().split())

def arg_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        if key == "tool":
            continue
        actual_value = actual.get(key)
        if key == "activity":
            a = set(norm(actual_value).split())
            e = set(norm(expected_value).split())
            if not e or len(a & e) / len(e) < 0.5:
                return False
        elif norm(actual_value) != norm(expected_value):
            return False
    return True

def evaluate_case(case: dict[str, Any], result: dict[str, Any], latency_ms: float) -> dict[str, Any]:
    calls = result.get("tool_calls") or []
    call_names = [str(c.get("name")) for c in calls]
    expected_tools = case["expected_tools"]
    required_args = case.get("required_args") or []
    matched_required = 0
    argument_correct = 0
    matched_indices: set[int] = set()
    for expected in required_args:
        tool = expected["tool"]
        found = next((i for i, call in enumerate(calls) if i not in matched_indices and call_names[i] == tool and arg_match(call.get("arguments") or {}, expected)), None)
        if found is not None:
            matched_required += 1
            argument_correct += 1
            matched_indices.add(found)
    if not required_args:
        matched_required = int(any(name in expected_tools for name in call_names))
        argument_correct = matched_required
    expected_count = len(required_args) if required_args else 1
    return {
        "id": case["id"], "category": case["category"], "question": case["question"], "success": bool(result.get("success")),
        "tool_selection_correct": matched_required == expected_count, "argument_accuracy": round(argument_correct / expected_count, 4),
        "required_tools_matched": matched_required, "required_tools": expected_tools, "actual_tools": call_names,
        "unnecessary_tool_calls": max(0, len(calls) - expected_count), "unexpected_tools": [name for name in call_names if name not in expected_tools],
        "rounds": result.get("rounds", 0), "latency_ms": round(latency_ms, 2), "trace_id": result.get("trace_id"),
    }

async def main() -> None:
    cases = json.loads(DATASET.read_text(encoding="utf-8"))["cases"]
    rows: list[dict[str, Any]] = []
    print(f"Running {len(cases)} end-to-end agent cases...\n")
    for index, case in enumerate(cases, 1):
        started = time.perf_counter()
        try:
            result = await run_agent(case["question"])
        except Exception as exc:
            result = {"success": False, "tool_calls": [], "rounds": 0, "trace_id": None, "error": str(exc)}
        row = evaluate_case(case, result, (time.perf_counter() - started) * 1000)
        row["error"] = result.get("error")
        rows.append(row)
        status = "PASS" if row["tool_selection_correct"] and row["argument_accuracy"] == 1 else "FAIL"
        print(f"[{index:02d}/{len(cases)}] {status:<4} {case['id']:<12} tools={row['actual_tools']} rounds={row['rounds']} latency={row['latency_ms']:.0f}ms")
    latencies = sorted(r["latency_ms"] for r in rows)
    summary = {
        "cases": len(rows), "tool_selection_accuracy": round(statistics.mean(r["tool_selection_correct"] for r in rows), 4),
        "argument_accuracy": round(statistics.mean(r["argument_accuracy"] for r in rows), 4), "agent_success_rate": round(statistics.mean(r["success"] for r in rows), 4),
        "unnecessary_tool_call_rate": round(sum(r["unnecessary_tool_calls"] for r in rows) / max(1, sum(len(r["actual_tools"]) for r in rows)), 4),
        "unexpected_tool_calls": sum(len(r["unexpected_tools"]) for r in rows), "mean_rounds": round(statistics.mean(r["rounds"] or 0 for r in rows), 3),
        "mean_latency_ms": round(statistics.mean(latencies), 2), "p50_latency_ms": round(statistics.median(latencies), 2),
        "p95_latency_ms": round(latencies[max(0, min(len(latencies) - 1, int(len(latencies) * 0.95) - 1))], 2),
    }
    category_metrics: dict[str, Any] = {}
    for category in sorted({r["category"] for r in rows}):
        subset = [r for r in rows if r["category"] == category]
        category_metrics[category] = {"cases": len(subset), "tool_selection_accuracy": round(statistics.mean(r["tool_selection_correct"] for r in subset), 4), "argument_accuracy": round(statistics.mean(r["argument_accuracy"] for r in subset), 4), "success_rate": round(statistics.mean(r["success"] for r in subset), 4)}
    report = {"dataset": DATASET.name, "summary": summary, "category_metrics": category_metrics, "failures": [r for r in rows if not r["tool_selection_correct"] or r["argument_accuracy"] < 1 or r["unexpected_tools"]], "rows": rows, "notes": ["Measures the actual Gemini + MCP agent loop, not a mocked planner.", "RAG cases accept either search_weather or ask_weather.", "Activity argument matching allows partial token overlap to avoid penalizing harmless phrasing differences.", "Knowledge Graph tools are not part of the allowed evaluation set."]}
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + "=" * 88)
    print("WEATHER MCP AGENT BENCHMARK")
    print("=" * 88)
    print(f"Cases: {summary['cases']}")
    print(f"Tool-selection accuracy : {summary['tool_selection_accuracy']:.1%}")
    print(f"Argument accuracy       : {summary['argument_accuracy']:.1%}")
    print(f"Agent success rate      : {summary['agent_success_rate']:.1%}")
    print(f"Unnecessary-call rate   : {summary['unnecessary_tool_call_rate']:.1%}")
    print(f"Unexpected tool calls   : {summary['unexpected_tool_calls']}")
    print(f"Mean rounds             : {summary['mean_rounds']:.2f}")
    print(f"Mean latency            : {summary['mean_latency_ms']:.1f} ms")
    print(f"P50 latency             : {summary['p50_latency_ms']:.1f} ms")
    print(f"P95 latency             : {summary['p95_latency_ms']:.1f} ms")
    print(f"Report: {REPORT}")

if __name__ == "__main__":
    asyncio.run(main())
