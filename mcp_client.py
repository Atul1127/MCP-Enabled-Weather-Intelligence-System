"""Reusable MCP stdio client for the Indian Weather Intelligence server."""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER_PATH = os.path.join(PROJECT_ROOT, "mcp_server.py")


def _python_executable() -> str:
    """Return the Python interpreter that has the project's dependencies."""
    configured = os.environ.get("WEATHER_PYTHON")
    if configured and os.path.isfile(configured):
        return configured

    current = os.path.abspath(sys.executable)
    if os.path.normcase(os.path.dirname(current)).endswith(
        os.path.normcase(os.path.join(".venv", "Scripts"))
    ):
        return current

    venv_python = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    if os.path.isfile(venv_python):
        return venv_python
    return current


@asynccontextmanager
async def connect(trace_id: str | None = None) -> AsyncIterator[ClientSession]:
    """Open an MCP stdio session using the project virtualenv.

    The server working directory and interpreter are explicit because Chainlit
    may be launched by a global Python installation. MCP stdout is reserved
    for JSON-RPC; startup diagnostics are therefore routed to stderr by the
    server.
    """
    python = _python_executable()
    server_env = os.environ.copy()
    server_env["WEATHER_PYTHON"] = python
    server_env["PYTHONUNBUFFERED"] = "1"
    if trace_id:
        server_env["WEATHER_TRACE_ID"] = trace_id

    server_params = StdioServerParameters(
        command=python,
        args=[SERVER_PATH],
        env=server_env,
        cwd=PROJECT_ROOT,
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    except Exception as exc:
        raise RuntimeError(
            f"MCP server failed to initialize using {python}: {exc}"
        ) from exc


def _tool_schema(tool: Any) -> dict[str, Any]:
    """Convert an MCP Tool definition into a provider-neutral function schema."""
    schema = getattr(tool, "input_schema", None)
    if schema is None:
        schema = getattr(tool, "inputSchema", None)
    if hasattr(schema, "model_dump"):
        schema = schema.model_dump(by_alias=True, exclude_none=True)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": schema or {"type": "object", "properties": {}},
        },
    }


async def discover_tools(session: ClientSession) -> list[dict[str, Any]]:
    """Discover MCP tools and return normalized function schemas."""
    response = await session.list_tools()
    tools = getattr(response, "tools", None)
    if tools is None:
        tools = []
        for item in response:
            if isinstance(item, tuple) and item and item[0] == "tools":
                tools.extend(item[1])
    return [_tool_schema(tool) for tool in tools]


async def call_tool(
    session: ClientSession,
    name: str,
    arguments: dict[str, Any] | None = None,
) -> Any:
    """Call an MCP tool through the live protocol session."""
    result = await session.call_tool(name, arguments=arguments or {})
    if result.structured_content is not None:
        return result.structured_content
    return {
        "content": [getattr(content, "text", str(content)) for content in result.content],
        "is_error": bool(getattr(result, "is_error", False)),
    }


async def main() -> None:
    """Smoke-test MCP discovery and one weather call."""
    async with connect() as session:
        tools = await discover_tools(session)
        print("Available MCP tools:")
        for tool in tools:
            print(f"- {tool['function']['name']}: {tool['function']['description']}")
        result = await call_tool(session, "get_weather", {"location": "Kolkata"})
        print("\nKolkata tool result:")
        print(result)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
