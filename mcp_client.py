"""MCP client for the Indian Weather Intelligence server.

This is the protocol boundary used by the future Ollama agent. It deliberately
connects to the server over stdio instead of importing the server module, which
keeps the MCP client/server boundary real and testable.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mcp_server.py",
)


async def list_tools() -> list[dict[str, Any]]:
    """Discover tools exposed by the weather MCP server."""

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_PATH],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            response = await session.list_tools()

            return [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in response.tools
            ]


async def call_tool(
    name: str,
    arguments: dict[str, Any] | None = None,
) -> Any:
    """Call one MCP tool through the real stdio protocol boundary."""

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_PATH],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                name,
                arguments=arguments or {},
            )

            if result.structured_content is not None:
                return result.structured_content

            return [
                getattr(content, "text", str(content))
                for content in result.content
            ]


async def main() -> None:
    """Smoke-test MCP discovery and one weather tool call."""

    tools = await list_tools()
    print("Available MCP tools:")
    for tool in tools:
        print(f"- {tool['name']}: {tool['description']}")

    result = await call_tool(
        "get_weather",
        {"location": "Kolkata"},
    )
    print("\nKolkata tool result:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
