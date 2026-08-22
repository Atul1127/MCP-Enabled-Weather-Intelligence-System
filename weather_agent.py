"""Local Ollama agent that dynamically orchestrates the weather MCP tools.

Architecture:

    User
      -> Ollama tool-calling model
      -> MCP client
      -> MCP stdio server
      -> weather / hybrid-RAG tools
      -> MCP result
      -> Ollama final answer

The LLM never imports weather code directly. It only sees MCP tool schemas
and invokes tools through the protocol boundary.

No paid LLM API is required.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

from ollama import AsyncClient

from mcp_client import call_tool, connect, discover_tools


OLLAMA_MODEL = os.environ.get(
    "WEATHER_AGENT_MODEL",
    "qwen3:4b",
)

MAX_TOOL_ROUNDS = int(
    os.environ.get(
        "WEATHER_AGENT_MAX_ROUNDS",
        "6",
    )
)

SYSTEM_PROMPT = """
You are the Indian Weather Intelligence Agent.

You have access to tools exposed through an MCP server. Use those tools when
real weather data or stored weather evidence is required.

Rules:
1. For current conditions or forecasts, use get_weather.
2. For questions requiring historical/stored weather evidence or semantic
   retrieval, use search_weather.
3. Do not invent weather observations, forecasts, warnings, or locations.
4. You may call multiple tools when a question requires comparison or more
   than one location.
5. Treat application severity as an analytical label, not an official IMD
   warning. Only call something an official IMD warning if the tool result
   explicitly identifies it as such.
6. Prefer concise, actionable answers with units in Celsius and km/h.
7. After tool results are available, synthesize the answer yourself. Do not
   expose internal tool-call JSON unless the user asks for it.
""".strip()


async def run_agent(query: str) -> str:
    """Run a bounded multi-step Ollama -> MCP agent loop."""

    if not query.strip():
        raise ValueError("Query cannot be empty")

    ollama = AsyncClient()

    async with connect() as session:
        tool_schemas = await discover_tools(session)

        # ask_weather is intentionally not exposed to the agent. It invokes an
        # LLM inside an MCP tool, which would create a nested model loop. The
        # agent should retrieve evidence through MCP and perform synthesis once.
        tool_schemas = [
            tool
            for tool in tool_schemas
            if tool["function"]["name"] in {
                "get_weather",
                "search_weather",
                "sync_weather",
                "database_health",
            }
        ]

        messages: list[Any] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": query.strip(),
            },
        ]

        for _ in range(MAX_TOOL_ROUNDS):
            response = await ollama.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                tools=tool_schemas,
                stream=False,
            )

            messages.append(response.message)
            tool_calls = response.message.tool_calls or []

            if not tool_calls:
                return (response.message.content or "").strip()

            for tool_call in tool_calls:
                name = tool_call.function.name
                arguments = dict(tool_call.function.arguments or {})

                if not any(
                    tool["function"]["name"] == name
                    for tool in tool_schemas
                ):
                    result: Any = {
                        "success": False,
                        "error": f"Tool '{name}' is not exposed by this agent.",
                    }
                else:
                    try:
                        result = await call_tool(
                            session,
                            name,
                            arguments,
                        )
                    except Exception as exc:
                        result = {
                            "success": False,
                            "error": str(exc),
                        }

                messages.append(
                    {
                        "role": "tool",
                        "tool_name": name,
                        "content": json.dumps(
                            result,
                            ensure_ascii=False,
                            default=str,
                        ),
                    }
                )

        return "I could not complete the weather analysis within the tool-call limit."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the local Indian Weather MCP agent."
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Natural-language weather question.",
    )
    args = parser.parse_args()

    query = " ".join(args.query).strip()

    if not query:
        query = input("Weather question: ").strip()

    print(asyncio.run(run_agent(query)))


if __name__ == "__main__":
    main()
