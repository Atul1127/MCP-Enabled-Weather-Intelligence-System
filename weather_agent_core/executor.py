"""Resilient MCP execution layer with policy, validation, timeout and bounded retries."""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from mcp_client import call_tool
from .security import validate_observation, validate_tool_arguments, validate_tool_call

MAX_FUNCTION_CALLS = max(1, int(os.environ.get("WEATHER_MAX_TOOL_CALLS", "8")))


class MCPExecutor:
    """Execute MCP calls through a policy and validation boundary."""

    def __init__(self, session: Any, allowed_tools: set[str], *, timeout_seconds: float | None = None, max_retries: int | None = None) -> None:
        self.session = session
        self.allowed_tools = frozenset(allowed_tools)
        self.timeout_seconds = float(os.environ.get("WEATHER_MCP_TIMEOUT", "20")) if timeout_seconds is None else float(timeout_seconds)
        self.max_retries = int(os.environ.get("WEATHER_MCP_RETRIES", "2")) if max_retries is None else int(max_retries)
        self._successful_results: dict[str, Any] = {}
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

    @staticmethod
    def _sanitize_result(result: Any) -> Any:
        """Do not feed raw remote exception text back into the model."""
        if isinstance(result, dict) and result.get("success") is False:
            return {
                "success": False,
                "error": "MCP tool reported an execution failure.",
                "error_type": str(result.get("error_type") or "mcp_error"),
            }
        return result

    @staticmethod
    def _key(name: str, args: dict[str, Any]) -> str:
        """Stable logical identity for one tool invocation."""
        return json.dumps([name, args], sort_keys=True, separators=(",", ":"), default=str)

    async def execute(self, function_calls: list[Any]) -> list[tuple[str, dict[str, Any], Any]]:
        if len(function_calls) > MAX_FUNCTION_CALLS:
            return [
                (
                    str(call.name),
                    dict(call.args or {}),
                    {"success": False, "error": f"Too many tool calls in one round; maximum is {MAX_FUNCTION_CALLS}.", "error_type": "policy_denied"},
                )
                for call in function_calls
            ]

        in_flight: dict[str, asyncio.Task[Any]] = {}

        async def one(call: Any) -> tuple[str, dict[str, Any], Any]:
            name, args = str(call.name), dict(call.args or {})
            if name not in self.allowed_tools:
                return name, args, {"success": False, "error": f"Tool '{name}' is not allowed.", "error_type": "policy_denied"}
            try:
                validate_tool_arguments(args)
                validate_tool_call(name, args)
                key = self._key(name, args)
                cached = self._successful_results.get(key)
                if cached is not None:
                    return name, args, cached

                task = in_flight.get(key)
                if task is None:
                    task = asyncio.create_task(self._call_with_retry(name, args))
                    in_flight[key] = task
                try:
                    result = await task
                finally:
                    if key in in_flight and in_flight[key].done():
                        in_flight.pop(key, None)

                validate_observation(result)
                result = self._sanitize_result(result)
                if not (isinstance(result, dict) and result.get("success") is False):
                    self._successful_results[key] = result
                return name, args, result
            except ValueError as exc:
                return name, args, {"success": False, "error": str(exc), "error_type": "validation_error"}
            except asyncio.TimeoutError:
                return name, args, {"success": False, "error": f"MCP tool '{name}' timed out after {self.timeout_seconds:g}s.", "error_type": "timeout"}
            except asyncio.CancelledError:
                raise
            except Exception:
                return name, args, {"success": False, "error": "MCP tool execution failed.", "error_type": "execution_error"}

        return await asyncio.gather(*(one(call) for call in function_calls))
