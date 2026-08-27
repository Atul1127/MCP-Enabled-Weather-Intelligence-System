"""Deterministic security gates for the agent and MCP boundaries."""
from __future__ import annotations

import re
from typing import Any

_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(all|any|the|previous)\s+instructions", re.I),
    re.compile(r"disregard\s+(all|any|the|previous)\s+instructions", re.I),
    re.compile(r"system\s+message\s*[:=]", re.I),
    re.compile(r"reveal\s+(the\s+)?(system|developer)\s+prompt", re.I),
)
_ALLOWED_ARG_TYPES = (str, int, float, bool, type(None), list, dict)


def inspect_text(text: str) -> dict[str, Any]:
    matches = [pattern.pattern for pattern in _INJECTION_PATTERNS if pattern.search(text)]
    return {"suspicious": bool(matches), "signals": matches}


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
