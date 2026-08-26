"""Lightweight local observability for the weather agent and RAG stack.

No external telemetry service is required. Events are emitted as JSON Lines so
runs can be inspected locally and shipped to another backend later.
"""

from __future__ import annotations

import contextvars
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

LOG_PATH = Path(os.environ.get("WEATHER_TRACE_PATH", "observability/traces.jsonl"))
_CURRENT_SPAN: contextvars.ContextVar[str | None] = contextvars.ContextVar("weather_current_span", default=None)


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def new_span_id() -> str:
    return uuid.uuid4().hex[:12]


def emit(event: str, *, trace_id: str, **fields: Any) -> None:
    payload = {
        "timestamp": time.time(),
        "event": event,
        "trace_id": trace_id,
        **fields,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


@contextmanager
def span(name: str, *, trace_id: str, **fields: Any) -> Iterator[dict[str, Any]]:
    started = time.perf_counter()
    span_id = new_span_id()
    parent_span_id = _CURRENT_SPAN.get()
    token = _CURRENT_SPAN.set(span_id)
    emit("span.start", trace_id=trace_id, span=name, span_id=span_id, parent_span_id=parent_span_id, **fields)
    result: dict[str, Any] = {}
    try:
        yield result
        result["ok"] = True
    except Exception as exc:
        result.update(ok=False, error=str(exc))
        raise
    finally:
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        emit("span.end", trace_id=trace_id, span=name, span_id=span_id, parent_span_id=parent_span_id, **result)
        _CURRENT_SPAN.reset(token)


def read_trace(trace_id: str) -> list[dict[str, Any]]:
    if not LOG_PATH.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("trace_id") == trace_id:
            events.append(item)
    return events


def summarize_trace(trace_id: str) -> dict[str, Any]:
    events = read_trace(trace_id)
    spans = [e for e in events if e.get("event") == "span.end"]
    tools = [e for e in events if e.get("event") == "agent.tool"]
    return {
        "trace_id": trace_id,
        "events": len(events),
        "tool_calls": len(tools),
        "tools": [e.get("tool") for e in tools],
        "spans": spans,
        "total_latency_ms": max((float(e.get("latency_ms", 0)) for e in spans), default=0.0),
    }
