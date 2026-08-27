"""Deterministic security and input-validation gates for agent/MCP boundaries."""
from __future__ import annotations

import re
from typing import Any

_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(all|any|the|previous)\s+instructions", re.I),
    re.compile(r"disregard\s+(all|any|the|previous)\s+instructions", re.I),
    re.compile(r"system\s+message\s*[:=]", re.I),
    re.compile(r"reveal\s+(the\s+)?(system|developer)\s+prompt", re.I),
    re.compile(r"(?:developer|system)\s+instructions?\s*[:=]", re.I),
    re.compile(r"jailbreak|prompt\s+injection", re.I),
)
_ALLOWED_ARG_TYPES = (str, int, float, bool, type(None), list, dict)
_MAX_LOCATION_LENGTH = 200
_MAX_QUERY_LENGTH = 5000
_MAX_ACTIVITY_LENGTH = 200
_MAX_TOP_K = 20
_MAX_SYNC_LOCATIONS = 50


def inspect_text(text: str) -> dict[str, Any]:
    """Detect common prompt-injection signals without treating the check as a guarantee."""
    if not isinstance(text, str):
        return {"suspicious": True, "signals": ["non_string_input"]}
    matches = [pattern.pattern for pattern in _INJECTION_PATTERNS if pattern.search(text)]
    return {"suspicious": bool(matches), "signals": matches}


def validate_user_query(query: str, *, max_length: int = _MAX_QUERY_LENGTH) -> str:
    """Validate a user-controlled query before it reaches retrieval or an LLM."""
    if not isinstance(query, str):
        raise ValueError("Query must be a string")
    text = query.strip()
    if not text:
        raise ValueError("Query cannot be empty")
    if len(text) > max_length:
        raise ValueError(f"Query exceeds maximum length of {max_length} characters")
    security = inspect_text(text)
    if security["suspicious"]:
        raise ValueError("Query contains a blocked prompt-injection signal")
    return text


def validate_location(location: str) -> str:
    if not isinstance(location, str):
        raise ValueError("location must be a string")
    value = location.strip()
    if not value:
        raise ValueError("location cannot be empty")
    if len(value) > _MAX_LOCATION_LENGTH:
        raise ValueError(f"location exceeds maximum length of {_MAX_LOCATION_LENGTH} characters")
    return value


def validate_activity(activity: str) -> str:
    if not isinstance(activity, str):
        raise ValueError("activity must be a string")
    value = activity.strip()
    if not value:
        raise ValueError("activity cannot be empty")
    if len(value) > _MAX_ACTIVITY_LENGTH:
        raise ValueError(f"activity exceeds maximum length of {_MAX_ACTIVITY_LENGTH} characters")
    return value


def validate_top_k(top_k: Any) -> int:
    if isinstance(top_k, bool):
        raise ValueError("top_k must be an integer")
    try:
        value = int(top_k)
    except (TypeError, ValueError) as exc:
        raise ValueError("top_k must be an integer") from exc
    if value < 1 or value > _MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {_MAX_TOP_K}")
    return value


def validate_tool_call(name: str, arguments: dict[str, Any]) -> None:
    """Apply tool-specific semantic limits after generic schema validation."""
    if name in {"get_weather", "get_forecast", "get_weather_alerts", "assess_weather_risk"}:
        arguments["location"] = validate_location(arguments.get("location"))
    if name in {"search_weather", "ask_weather"}:
        arguments["query"] = validate_user_query(arguments.get("query"))
        if "top_k" in arguments:
            arguments["top_k"] = validate_top_k(arguments["top_k"])
        if arguments.get("location") is not None:
            arguments["location"] = validate_location(arguments["location"])
        if arguments.get("state") is not None:
            arguments["state"] = validate_location(arguments["state"])
    if name == "assess_weather_risk" and arguments.get("activity") is not None:
        arguments["activity"] = validate_activity(arguments["activity"])
    if name == "sync_weather":
        locations = arguments.get("locations")
        if not isinstance(locations, list) or not locations:
            raise ValueError("locations must be a non-empty list")
        if len(locations) > _MAX_SYNC_LOCATIONS:
            raise ValueError(f"locations cannot contain more than {_MAX_SYNC_LOCATIONS} items")
        arguments["locations"] = [validate_location(item) for item in locations]


def validate_tool_arguments(arguments: dict[str, Any], *, max_depth: int = 5) -> None:
    """Reject malformed or unbounded tool arguments before an MCP call."""
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object")

    def walk(value: Any, depth: int) -> None:
        if depth > max_depth:
            raise ValueError("Tool arguments exceed maximum nesting depth")
        if not isinstance(value, _ALLOWED_ARG_TYPES):
            raise ValueError(f"Unsupported tool argument type: {type(value).__name__}")
        if isinstance(value, str) and len(value) > 10000:
            raise ValueError("Tool argument string exceeds maximum length")
        if isinstance(value, list):
            if len(value) > 1000:
                raise ValueError("Tool argument list is too large")
            for item in value:
                walk(item, depth + 1)
        elif isinstance(value, dict):
            if len(value) > 100:
                raise ValueError("Tool argument object has too many fields")
            for key, item in value.items():
                if not isinstance(key, str) or len(key) > 200:
                    raise ValueError("Invalid tool argument key")
                walk(item, depth + 1)

    walk(arguments, 0)


def validate_observation(result: Any, *, max_depth: int = 6, max_text: int = 50000) -> Any:
    """Reject oversized or excessively nested untrusted MCP output."""
    def walk(value: Any, depth: int) -> None:
        if depth > max_depth:
            raise ValueError("MCP result exceeds maximum nesting depth")
        if isinstance(value, str):
            if len(value) > max_text:
                raise ValueError("MCP result contains oversized text")
            return
        if isinstance(value, list):
            if len(value) > 2000:
                raise ValueError("MCP result list is too large")
            for item in value:
                walk(item, depth + 1)
            return
        if isinstance(value, dict):
            if len(value) > 200:
                raise ValueError("MCP result object has too many fields")
            for key, item in value.items():
                if not isinstance(key, str) or len(key) > 200:
                    raise ValueError("MCP result contains an invalid key")
                walk(item, depth + 1)
            return
        if not isinstance(value, _ALLOWED_ARG_TYPES):
            raise ValueError(f"Unsupported MCP result type: {type(value).__name__}")

    walk(result, 0)
    return result
