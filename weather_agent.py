"""Fast local Ollama agent for the Indian Weather MCP server.

Architecture:
    User -> Ollama tool selection -> MCP client -> MCP server -> tool result
          -> Ollama final answer

The agent intentionally keeps the loop small: one planning/tool round and one
final synthesis round for normal weather questions. This keeps local inference
fast while preserving real MCP orchestration.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

from ollama import AsyncClient

from mcp_client import call_tool, connect, discover_tools


OLLAMA_MODEL = os.environ.get("WEATHER_AGENT_MODEL", "llama3.2:3b")
MAX_TOOL_ROUNDS = int(os.environ.get("WEATHER_AGENT_MAX_ROUNDS", "2"))

SYSTEM_PROMPT = """
You are an Indian Weather Intelligence Agent.
Use MCP tools when live weather data, weather risk, forecast hazards, or weather
knowledge is required. Select the most specific tool for the user's intent.

Routing rules:
- get_weather: current conditions or forecast values for a location.
- assess_weather_risk: decide whether a specific activity is safe/suitable,
  including questions such as "Is it good for cricket?", "Should I hike?",
  or "Is outdoor activity safe?" It computes a practical risk score.
- get_weather_alerts: find forecast hazards or alerts such as heavy rain,
  thunderstorms, extreme heat, or strong wind. Use it when the user asks about
  dangers, hazards, alerts, warnings, or what dangerous weather is expected.
- search_weather: answer general conceptual/knowledge questions about weather,
  weather-risk concepts, or stored weather guidance. If the user asks what,
  why, or how a weather concept works without asking for a location-specific
  safety decision, prefer search_weather over assess_weather_risk.

Do not use assess_weather_risk merely because the question contains the word
"risk". Use it when the user wants a location/activity-specific decision.
You may call multiple tools for comparisons. Never invent weather data.
Give concise, actionable answers in Celsius and km/h.
""".strip()

AGENT_TOOLS = {
    "get_weather",
    "assess_weather_risk",
    "get_weather_alerts",
    "search_weather",
}


async def run_agent(query: str) -> str:
    """Run a bounded Ollama -> MCP agent loop."""
    if not query.strip():
        raise ValueError("Query cannot be empty")

    ollama = AsyncClient()

    async with connect() as session:
        tool_schemas = [
            tool
            for tool in await discover_tools(session)
            if tool["function"]["name"] in AGENT_TOOLS
        ]

        messages: list[Any] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query.strip()},
        ]

        for _ in range(MAX_TOOL_ROUNDS):
            response = await ollama.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                tools=tool_schemas,
                stream=False,
                keep_alive="10m",
                options={"temperature": 0},
            )

            messages.append(response.message)
            tool_calls = response.message.tool_calls or []

            if not tool_calls:
                return (response.message.content or "").strip()

            async def execute(tool_call: Any) -> tuple[str, Any]:
                name = tool_call.function.name
                arguments = dict(tool_call.function.arguments or {})

                if name not in AGENT_TOOLS:
                    return name, {
                        "success": False,
                        "error": f"Tool '{name}' is not exposed by this agent.",
                    }

                try:
                    return name, await call_tool(session, name, arguments)
                except Exception as exc:
                    return name, {"success": False, "error": str(exc)}

            results = await asyncio.gather(
                *(execute(tool_call) for tool_call in tool_calls)
            )

            for name, result in results:
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
        description="Run the fast local Indian Weather MCP agent."
    )
    parser.add_argument("query", nargs="*", help="Natural-language weather question.")
    args = parser.parse_args()

    query = " ".join(args.query).strip()
    if not query:
        query = input("Weather question: ").strip()

    print(asyncio.run(run_agent(query)))


if __name__ == "__main__":
    main()
