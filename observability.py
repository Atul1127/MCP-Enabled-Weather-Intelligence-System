"""Lightweight structured observability for the MCP weather agent."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

LOGGER = logging.getLogger("weather_agent")


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


@contextmanager
def measure(operation: str, request_id: str | None = None, **metadata: Any) -> Iterator[dict[str, Any]]:
    started = time.perf_counter()
    record: dict[str, Any] = {
        "request_id": request_id or new_request_id(),
        "operation": operation,
        **metadata,
    }
    try:
        yield record
        record["success"] = True
    except Exception as exc:
        record["success"] = False
        record["error"] = str(exc)
        raise
    finally:
        record["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        LOGGER.info("weather_observation %s", json.dumps(record, ensure_ascii=False, default=str))
