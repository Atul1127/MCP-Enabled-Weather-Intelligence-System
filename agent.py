"""Explicit local ReAct-style weather agent.

Architecture:
  observe -> plan/tool selection -> execute MCP tools -> observe results ->
  decide whether more evidence is needed -> synthesize grounded response.

The agent is intentionally provider-local: Ollama performs reasoning and the
existing MCP server owns tool execution.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from ollama import AsyncClient

from mcp_client import call_tool, connect, discover_tools
from observability import emit, new_trace_id, span

MODEL = os.environ.get("WEATHER_AGENT_MODEL", "llama3.2:3b")
MAX_ROUNDS = int(os.environ.get("WEATHER_AGENT_MAX_ROUNDS", "4"))
ALLOWED_TOOLS = {
    "get_weather",
    "get_weather_alerts",
    "assess_weather_risk",
    "search_weather",
    "ask_weather",
}

SYSTEM_PROMPT = """You are an Indian Weather Intelligence Agent.
You are an orchestrator, not a database. Use MCP tools for evidence and never
invent weather facts.

Choose tools deliberately:
- get_weather for current conditions and forecasts.
- get_weather_alerts for hazards and forecast danger signals.
- assess_weather_risk for activity-specific suitability.
- search_weather for evidence retrieval and historical/conceptual questions.
- ask_weather when a grounded RAG answer is directly useful.

For comparisons, call the relevant tool for every location. You may make
multiple tool calls in one turn. After observing tool results, decide whether
another tool is required. Finish only when the evidence is sufficient.
Keep the final answer concise, distinguish live forecast data from historical
knowledge, and preserve citations returned by RAG tools."""


@dataclass
class AgentState:
    trace_id: str
    query: str
    round: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)


async def run_agent(query: str) -> dict[str, Any]:
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty")

    state = AgentState(trace_id=new_trace_id(), query=query)
    started = time.perf_counter()
    emit("agent.start", trace_id=state.trace_id, query=query, model=MODEL)
    ollama = AsyncClient()

    async with connect() as session:
        discovered = await discover_tools(session)
        tools = [t for t in discovered if t["function"]["name"] in ALLOWED_TOOLS]
        messages: list[Any] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        for round_no in range(1, MAX_ROUNDS + 1):
            state.round = round_no
            with span("agent.reason", trace_id=state.trace_id, round=round_no) as info:
                response = await ollama.chat(
                    model=MODEL,
                    messages=messages,
                    tools=tools,
                    stream=False,
                    keep_alive="10m",
                    options={"temperature": 0},
                )
                tool_calls = response.message.tool_calls or []
                info["tool_calls"] = len(tool_calls)

            messages.append(response.message)
            if not tool_calls:
                answer = (response.message.content or "").strip()
                emit("agent.end", trace_id=state.trace_id, rounds=round_no, tools=len(state.tool_calls), latency_ms=round((time.perf_counter() - started) * 1000, 2))
                return {
                    "success": True,
                    "answer": answer,
                    "trace_id": state.trace_id,
                    "rounds": round_no,
                    "tool_calls": state.tool_calls,
                    "observations": state.observations,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                }

            async def execute(call: Any) -> tuple[str, dict[str, Any], Any]:
                name = call.function.name
                arguments = dict(call.function.arguments or {})
                if name not in ALLOWED_TOOLS:
                    result = {"success": False, "error": f"Tool '{name}' is not allowed."}
                else:
                    try:
                        result = await call_tool(session, name, arguments)
                    except Exception as exc:
                        result = {"success": False, "error": str(exc)}
                return name, arguments, result

            with span("agent.execute_tools", trace_id=state.trace_id, round=round_no) as info:
                results = await asyncio.gather(*(execute(c) for c in tool_calls))
                info["executed"] = len(results)

            for name, arguments, result in results:
                state.tool_calls.append({"name": name, "arguments": arguments})
                observation = {"tool": name, "result": result}
                state.observations.append(observation)
                emit("agent.tool", trace_id=state.trace_id, tool=name, arguments=arguments, success=result.get("success", True) if isinstance(result, dict) else True)
                messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })

        emit("agent.max_rounds", trace_id=state.trace_id, rounds=MAX_ROUNDS)
        return {
            "success": False,
            "answer": "I could not gather enough evidence within the agent round limit.",
            "trace_id": state.trace_id,
            "rounds": MAX_ROUNDS,
            "tool_calls": state.tool_calls,
            "observations": state.observations,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local MCP weather agent.")
    parser.add_argument("query", nargs="*", help="Natural-language weather question")
    args = parser.parse_args()
    query = " ".join(args.query).strip() or input("Weather question: ").strip()
    result = asyncio.run(run_agent(query))
    print(result["answer"])
    print(f"\ntrace_id={result['trace_id']} rounds={result['rounds']}")


if __name__ == "__main__":
    main()
