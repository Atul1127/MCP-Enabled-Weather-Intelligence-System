"""Explicit local ReAct-style weather agent.

Architecture:
  observe -> plan/tool selection -> execute MCP tools -> observe results ->
  decide whether more evidence is needed -> synthesize grounded response.

The agent is intentionally provider-local: Ollama performs reasoning and the
existing MCP server owns tool execution. Simple current-condition questions
use deterministic rendering after the tool call so the LLM cannot substitute
forecast values or hallucinate the current sky condition/time of day.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
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

For current-condition questions, current_summary returned by get_weather is
authoritative. It contains the live observation time, timezone, time of day,
weather condition, temperature, humidity, cloud cover and wind. NEVER use a
value from the daily forecast to describe the current condition. NEVER infer
day/night or sky condition from temperature, precipitation probability or
forecast values. If current_summary and another field appear inconsistent,
current_summary wins.

For comparisons, call the relevant tool for every location. You may make
multiple tool calls in one turn. After observing tool results, decide whether
another tool is required. Finish only when the evidence is sufficient.
Keep the final answer concise, distinguish live observation/forecast data from
historical knowledge, and preserve citations returned by RAG tools."""


@dataclass
class AgentState:
    trace_id: str
    query: str
    round: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)


def _is_simple_current_weather_query(query: str) -> bool:
    """Detect queries whose answer should come directly from live conditions."""
    text = query.lower().strip()
    if any(token in text for token in ("tomorrow", "forecast", "next week", "this week")):
        return False
    patterns = (
        r"\bcurrent weather\b",
        r"\bcurrent conditions?\b",
        r"\bweather right now\b",
        r"\bweather now\b",
        r"\bwhat(?:'s| is) the weather\b",
        r"\bhow is the weather\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _render_current_weather(result: dict[str, Any]) -> str | None:
    """Render live current conditions without asking the LLM to reinterpret them."""
    summary = result.get("current_summary") if isinstance(result, dict) else None
    if not isinstance(summary, dict):
        return None
    location = result.get("location", {})
    display_name = location.get("display_name") or "the requested location"
    time_of_day = summary.get("time_of_day")
    condition = summary.get("condition")
    temperature = summary.get("temperature_c")
    feels_like = summary.get("apparent_temperature_c")
    humidity = summary.get("relative_humidity_pct")
    cloud = summary.get("cloud_cover_pct")
    wind = summary.get("wind_speed_kmh")
    observation_time = summary.get("observation_time")
    timezone = summary.get("timezone")

    parts = [f"Current weather in {display_name}:"]
    if time_of_day in {"day", "night"}:
        parts.append(f"it is currently {time_of_day}")
    if condition:
        parts.append(f"with {condition.lower()}")
    if temperature is not None:
        parts.append(f"and {temperature}°C")
    if feels_like is not None:
        parts.append(f"(feels like {feels_like}°C)")
    if humidity is not None:
        parts.append(f"Humidity is {humidity}%.")
    if cloud is not None:
        parts.append(f"Cloud cover is {cloud}%.")
    if wind is not None:
        parts.append(f"Wind is {wind} km/h.")
    answer = " ".join(parts)
    if observation_time:
        answer += f" Observation: {observation_time}"
        if timezone:
            answer += f" ({timezone})."
        else:
            answer += "."
    return answer


async def run_agent(query: str) -> dict[str, Any]:
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty")

    state = AgentState(trace_id=new_trace_id(), query=query)
    started = time.perf_counter()
    emit("agent.start", trace_id=state.trace_id, query=query, model=MODEL)
    ollama = AsyncClient()
    simple_current_query = _is_simple_current_weather_query(query)

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

                if simple_current_query and name == "get_weather" and isinstance(result, dict) and result.get("success"):
                    deterministic_answer = _render_current_weather(result)
                    if deterministic_answer:
                        emit("agent.deterministic_current_answer", trace_id=state.trace_id)
                        emit("agent.end", trace_id=state.trace_id, rounds=round_no, tools=len(state.tool_calls), latency_ms=round((time.perf_counter() - started) * 1000, 2), deterministic=True)
                        return {
                            "success": True,
                            "answer": deterministic_answer,
                            "trace_id": state.trace_id,
                            "rounds": round_no,
                            "tool_calls": state.tool_calls,
                            "observations": state.observations,
                            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                            "deterministic": True,
                        }

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
