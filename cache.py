"""Optional Redis-backed cache with a safe in-memory fallback."""

from __future__ import annotations

import json
import os
import time
from typing import Any

try:
    import redis
except ImportError:  # pragma: no cover
    redis = None


class WeatherCache:
    def __init__(self) -> None:
        self.url = os.environ.get("REDIS_URL")
        self._memory: dict[str, tuple[float, Any]] = {}
        self._client = None
        if self.url and redis is not None:
            self._client = redis.Redis.from_url(self.url, decode_responses=True)

    def get(self, key: str) -> Any | None:
        if self._client is not None:
            value = self._client.get(key)
            return json.loads(value) if value else None
        item = self._memory.get(key)
        if not item or item[0] <= time.time():
            self._memory.pop(key, None)
            return None
        return item[1]

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        if self._client is not None:
            self._client.setex(key, ttl_seconds, json.dumps(value, default=str))
            return
        self._memory[key] = (time.time() + ttl_seconds, value)


cache = WeatherCache()


def weather_cache_key(latitude: float, longitude: float) -> str:
    return f"weather:{latitude:.4f}:{longitude:.4f}"


def geocode_cache_key(location: str) -> str:
    return f"geocode:{location.strip().lower()}"
