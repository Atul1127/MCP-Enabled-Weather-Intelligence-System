"""Summarize local Weather MCP traces.

Run from repository root:
    python evaluation/trace_report.py <trace_id>

No external telemetry service is required.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from observability import read_trace, summarize_trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one local Weather MCP trace")
    parser.add_argument("trace_id")
    args = parser.parse_args()

    events = read_trace(args.trace_id)
    summary = summarize_trace(args.trace_id)
    if not events:
        print(f"Trace not found: {args.trace_id}")
        raise SystemExit(1)

    span_events = [e for e in events if e.get("event") == "span.end"]
    tool_events = [e for e in events if e.get("event") == "agent.tool"]
    print("WEATHER MCP TRACE")
    print("=" * 72)
    print(f"Trace ID       : {args.trace_id}")
    print(f"Events         : {len(events)}")
    print(f"Tool calls     : {len(tool_events)}")
    print(f"Tools          : {[e.get('tool') for e in tool_events]}")
    print("\nSPANS")
    print("-" * 72)
    for event in span_events:
        parent = event.get("parent_span_id") or "-"
        print(f"{event.get('span','?'):<24} {float(event.get('latency_ms',0)):>9.1f} ms  parent={parent}")

    latencies = [float(e.get("latency_ms", 0)) for e in span_events]
    if latencies:
        print("\nSPAN LATENCY")
        print("-" * 72)
        print(f"Mean : {statistics.mean(latencies):.1f} ms")
        print(f"P50  : {statistics.median(latencies):.1f} ms")
        print(f"Max  : {max(latencies):.1f} ms")

    print("\nJSON SUMMARY")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
