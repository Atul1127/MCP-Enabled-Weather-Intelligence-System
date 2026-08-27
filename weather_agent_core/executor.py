"""Resilient MCP execution layer with policy, validation, timeout and bounded retries."""
from __future__ import annotations

import asyncio
import os
from typing import Any

from mcp_client import call_tool
from .security import validate_observation, validate_tool_arguments, validate_tool_call

MAX_FUNCTION_CALLS = max(1, int(os.environ.get("WEATHER_MAX_TOOL_CALLS", "16")))


class MCPExecutor:
    """Execute MCP calls through a policy and validation boundary."""

    def __init__(self, session: Any, allowed_tools: set[str], *, timeout_seconds: float | None = None, max_retries: int | None = None) -> None:
        self.session = session
        self.allowed_tools = frozenset(allowed_tools)
        self.timeout_seconds = float(os.environ.get("WEATHER_MCP_TIMEOUT", "20")) if timeout_seconds is None else float(timeout_seconds)
        self.max_retries = int(os.environ.get("WEATHER_MCP_RETRIES", "2")) if max_retries is None else int(max_retries)
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")

    async def _call_with_retry(self, name: str, args: dict[str, Any]) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await asyncio.wait_for(call_tool(self.session, name, args), timeout=self.timeout_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(min(0.25 * (2**attempt), 2.0))
        assert last_error is not None
        raise last_error

    async def execute(self, function_calls: list[Any]) -> list[tuple[str, dict[str, Any], Any]]:
        if len(function_calls) > MAX_FUNCTION_CALLS:
            return [
                ("__batch__", {}, {
                    "success": False,
                    "error": f"Too many tool calls in one round; maximum is {MAX_FUNCTION_CALLS}.",
                    "error_type": "policy_denied",
                })
            ]

        async def one(call: Any) -> tuple[str, dict[str, Any], Any]:
            name, args = str(call.name), dict(call.args or {})
            if name not in self.allowed_tools:
                return name, args, {"success": False, "error": f"Tool '{name}' is not allowed.", "error_type": "policy_denied"}
            try:
                validate_tool_arguments(args)
                validate_tool_call(name, args)
                result = await self._call_with_retry(name, args)
                validate_observation(result)
                return name, args, result
            except ValueError as exc:
                return name, args, {"success": False, "error": str(exc), "error_type": "validation_error"}
            except asyncio.TimeoutError:
                return name, args, {"success": False, "error": f"MCP tool '{name}' timed out after {self.timeout_seconds:g}s.", "error_type": "timeout"}
            except Exception as exc:
                return name, args, {"success": False, "error": str(exc), "error_type": "execution_error"}

        return await asyncio.gather(*(one(call) for call in function_calls))
