"""Local ReAct-style weather agent with MCP tools and grounded responses."""
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
    "get_weather", "get_forecast", "get_weather_alerts",
    "assess_weather_risk", "search_weather", "ask_weather",
}

SYSTEM_PROMPT = """You are an Indian Weather Intelligence Agent and MCP orchestrator.
Use tools for evidence and never invent weather facts.

- get_weather: current conditions plus raw 7-day forecast.
- get_forecast: REQUIRED for a specific future day. Pass date='tomorrow' or YYYY-MM-DD.
- get_weather_alerts: hazards across the forecast window.
- assess_weather_risk: activity suitability. Pass date='tomorrow' for tomorrow questions.
- search_weather / ask_weather: grounded retrieval for historical/conceptual questions.

Current conditions: current_summary from get_weather is authoritative. Never use daily
forecast fields for current sky condition, temperature, or time of day.
Future conditions: never use current fields as tomorrow's forecast.
Comparisons: call forecast/risk tools separately for every location. Do not finalize a
comparison after receiving only the first location's result; collect all requested
locations before answering.
RAG failure: if search_weather/ask_weather fails or returns success=false, do not answer
from general model knowledge; report grounded retrieval is unavailable.
RAG citations: when using search_weather/ask_weather, preserve [S1], [S2], etc. from the
returned evidence. Never invent citation IDs. Include a Sources section when RAG evidence
was used.
Keep answers concise and distinguish live observations, forecasts, and retrieved knowledge."""


@dataclass
class AgentState:
    trace_id: str
    query: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)


def _is_simple_current_weather_query(query: str) -> bool:
    text = query.lower().strip()
    if any(x in text for x in ("tomorrow", "forecast", "next week", "this week")):
        return False
    patterns = (
        r"\bcurrent weather\b", r"\bcurrent conditions?\b",
        r"\bweather right now\b", r"\bweather now\b",
        r"\bwhat(?:'s| is) the weather\b", r"\bhow is the weather\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _is_comparison_query(query: str) -> bool:
    text = query.lower().strip()
    return any(
        phrase in text
        for phrase in (
            "compare ",
            "compare the ",
            "which is better",
            "which would be better",
            "between ",
            "versus ",
            " vs ",
        )
    )


def _render_current_weather(result: dict[str, Any]) -> str | None:
    summary = result.get("current_summary")
    if not isinstance(summary, dict):
        return None
    name = result.get("location", {}).get("display_name") or "the requested location"
    parts = [f"Current weather in {name}:"]
    if summary.get("time_of_day") in {"day", "night"}:
        parts.append(f"it is currently {summary['time_of_day']}")
    if summary.get("condition"):
        parts.append(f"with {summary['condition'].lower()}")
    if summary.get("temperature_c") is not None:
        parts.append(f"and {summary['temperature_c']}°C")
    if summary.get("apparent_temperature_c") is not None:
        parts.append(f"(feels like {summary['apparent_temperature_c']}°C)")
    if summary.get("relative_humidity_pct") is not None:
        parts.append(f"Humidity is {summary['relative_humidity_pct']}%.")
    if summary.get("cloud_cover_pct") is not None:
        parts.append(f"Cloud cover is {summary['cloud_cover_pct']}%.")
    if summary.get("wind_speed_kmh") is not None:
        parts.append(f"Wind is {summary['wind_speed_kmh']} km/h.")
    if summary.get("observation_time"):
        parts.append(f"Observation: {summary['observation_time']} ({summary.get('timezone')}).")
    return " ".join(parts)


def _render_forecast(result: dict[str, Any]) -> str | None:
    forecast = result.get("forecast")
    if not isinstance(forecast, dict):
        return None
    name = result.get("location", {}).get("display_name", "the requested location")
    return (
        f"Forecast for {name} on {forecast.get('date')}: {forecast.get('condition')}, "
        f"{forecast.get('temperature_min_c')}–{forecast.get('temperature_max_c')}°C, "
        f"{forecast.get('precipitation_probability_pct')}% precipitation probability, "
        f"{forecast.get('precipitation_mm')} mm precipitation, and maximum wind "
        f"{forecast.get('max_wind_kmh')} km/h."
    )


def _rag_sources(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for observation in observations:
        if observation.get("tool") not in {"search_weather", "ask_weather"}:
            continue
        result = observation.get("result")
        if not isinstance(result, dict) or not result.get("success"):
            continue
        for source in result.get("sources") or []:
            if not isinstance(source, dict):
                continue
            citation = str(source.get("citation") or "").strip()
            if not citation or citation in seen:
                continue
            seen.add(citation)
            sources.append(source)
    return sources


def _append_rag_sources(answer: str, observations: list[dict[str, Any]]) -> str:
    sources = _rag_sources(observations)
    if not sources:
        return answer
    if re.search(r"(?im)^\s*sources\s*:?\s*$", answer):
        return answer
    lines = ["", "Sources:"]
    for source in sources:
        citation = source.get("citation", "")
        title = source.get("title") or source.get("source") or "Local weather knowledge base"
        topic = source.get("topic")
        suffix = f" — {topic}" if topic else ""
        lines.append(f"- [{citation}] {title}{suffix}")
    return answer.rstrip() + "\n" + "\n".join(lines)


async def run_agent(query: str) -> dict[str, Any]:
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty")

    state = AgentState(trace_id=new_trace_id(), query=query)
    started = time.perf_counter()
    emit("agent.start", trace_id=state.trace_id, query=query, model=MODEL)
    ollama = AsyncClient()
    simple_current = _is_simple_current_weather_query(query)
    comparison_query = _is_comparison_query(query)
    retrieval_failure = False

    async with connect() as session:
        discovered = await discover_tools(session)
        tools = [t for t in discovered if t["function"]["name"] in ALLOWED_TOOLS]
        messages: list[Any] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        for round_no in range(1, MAX_ROUNDS + 1):
            with span("agent.reason", trace_id=state.trace_id, round=round_no) as info:
                response = await ollama.chat(
                    model=MODEL,
                    messages=messages,
                    tools=tools,
                    stream=False,
                    keep_alive="10m",
                    options={"temperature": 0},
                )
                calls = response.message.tool_calls or []
                info["tool_calls"] = len(calls)

            messages.append(response.message)
            if not calls:
                if retrieval_failure:
                    answer = "I could not produce a grounded answer because the weather knowledge retrieval service failed. Please retry after the retrieval service is available."
                else:
                    answer = (response.message.content or "").strip()
                    answer = _append_rag_sources(answer, state.observations)
                emit(
                    "agent.end",
                    trace_id=state.trace_id,
                    rounds=round_no,
                    tools=len(state.tool_calls),
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                )
                return {
                    "success": not retrieval_failure,
                    "answer": answer,
                    "trace_id": state.trace_id,
                    "rounds": round_no,
                    "tool_calls": state.tool_calls,
                    "observations": state.observations,
                    "sources": _rag_sources(state.observations),
                }

            async def execute(call: Any):
                name = call.function.name
                args = dict(call.function.arguments or {})
                try:
                    result = await call_tool(session, name, args) if name in ALLOWED_TOOLS else {
                        "success": False,
                        "error": f"Tool '{name}' is not allowed.",
                    }
                except Exception as exc:
                    result = {"success": False, "error": str(exc)}
                return name, args, result

            with span("agent.execute_tools", trace_id=state.trace_id, round=round_no) as info:
                results = await asyncio.gather(*(execute(call) for call in calls))
                info["executed"] = len(results)

            for name, args, result in results:
                state.tool_calls.append({"name": name, "arguments": args})
                state.observations.append({"tool": name, "result": result})
                success = result.get("success", True) if isinstance(result, dict) else True
                emit("agent.tool", trace_id=state.trace_id, tool=name, arguments=args, success=success)

                if name in {"search_weather", "ask_weather"} and not success:
                    retrieval_failure = True

                messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })

                if simple_current and name == "get_weather" and isinstance(result, dict) and result.get("success"):
                    answer = _render_current_weather(result)
                    if answer:
                        return {
                            "success": True,
                            "answer": answer,
                            "trace_id": state.trace_id,
                            "rounds": round_no,
                            "tool_calls": state.tool_calls,
                            "observations": state.observations,
                            "deterministic": True,
                        }

                if (
                    not comparison_query
                    and name == "get_forecast"
                    and isinstance(result, dict)
                    and result.get("success")
                    and any(x in query.lower() for x in ("tomorrow", "forecast", "weather be like"))
                ):
                    answer = _render_forecast(result)
                    if answer:
                        return {
                            "success": True,
                            "answer": answer,
                            "trace_id": state.trace_id,
                            "rounds": round_no,
                            "tool_calls": state.tool_calls,
                            "observations": state.observations,
                            "deterministic": True,
                        }

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
    parser.add_argument("query", nargs="*")
    args = parser.parse_args()
    prompt = " ".join(args.query).strip() or input("Weather question: ").strip()
    result = asyncio.run(run_agent(prompt))
    print(result["answer"])
    print(f"\ntrace_id={result['trace_id']} rounds={result['rounds']}")


if __name__ == "__main__":
    main()
