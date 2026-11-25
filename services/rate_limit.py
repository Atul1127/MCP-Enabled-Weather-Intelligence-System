"""Small per-client rate limiter for the public endpoints that call Gemini.

The limit is kept in memory, so it applies per worker process. That is enough
to stop one visitor from draining the Gemini quota on a demo deployment; a
multi-instance deployment should use a shared store such as Redis instead.

Configure with, for example:
    WEATHER_AGENT_RATE_LIMIT=10/minute
    WEATHER_ASK_RATE_LIMIT=20/minute
    WEATHER_TRUST_PROXY=1   # behind Railway/NGINX: use X-Forwarded-For
Set a limit to "0" to disable it.
"""
from __future__ import annotations

import math
import os
import threading
import time
from collections import defaultdict, deque
from functools import wraps
from typing import Any, Callable

from flask import jsonify, request

_PERIODS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}


def parse_limit(spec: str) -> tuple[int, int] | None:
    """Parse "10/minute" into (10, 60). Returns None when disabled."""
    spec = (spec or "").strip().lower()
    if not spec or spec == "0":
        return None
    count, _, period = spec.partition("/")
    seconds = _PERIODS.get(period.strip().rstrip("s"), None)
    if seconds is None:
        raise ValueError(f"Invalid rate limit {spec!r}; use e.g. '10/minute'")
    return int(count), seconds


class SlidingWindowLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window_seconds: int) -> float:
        """Record a hit. Returns 0 if allowed, else seconds until a slot frees."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= now - window_seconds:
                hits.popleft()
            if len(hits) >= limit:
                return max(0.0, hits[0] + window_seconds - now)
            hits.append(now)
            return 0.0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


_LIMITER = SlidingWindowLimiter()


def client_key() -> str:
    if os.environ.get("WEATHER_TRUST_PROXY", "0") == "1":
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def rate_limited(env_var: str, default: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: reject with HTTP 429 when a client exceeds the limit."""

    def decorator(view: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(view)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            parsed = parse_limit(os.environ.get(env_var, default))
            if parsed is not None:
                limit, window = parsed
                wait = _LIMITER.check(f"{view.__name__}:{client_key()}", limit, window)
                if wait > 0:
                    response = jsonify({"error": "Too many requests; please try again shortly."})
                    response.status_code = 429
                    response.headers["Retry-After"] = str(math.ceil(wait))
                    return response
            return view(*args, **kwargs)

        return wrapper

    return decorator


def reset() -> None:
    _LIMITER.reset()
