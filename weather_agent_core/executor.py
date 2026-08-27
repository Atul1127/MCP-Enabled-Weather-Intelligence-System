"""MCP execution layer: the agent never imports weather capabilities directly."""
from __future__ import annotations

import asyncio
from typing import Any

from mcp_client import call_tool


class MCPExecutor:
    def __init__(self, session: Any, allowed_tools: set[str]):
        self.session = session
        self.allowed_tools = allowed_tools

    async def execute(self, function_calls: list[Any]) -> list[tuple[str, dict[str, Any], Any]]:
        async def one(call: Any) -> tuple[str, dict[str, Any], Any]:
            name = call.name
            args = dict(call.args or {})
            if name not in self.allowed_tools:
                return name, args, {"success": False, "error": f"Tool '{name}' is not allowed."}
            try:
                result = await call_tool(self.session, name, args)
                return name, args, result
            except Exception as exc:
                return name, args, {"success": False, "error": str(exc)}

        return await asyncio.gather(*(one(call) for call in function_calls))
