"""Gemini ReAct-style weather agent with MCP tools and grounded responses."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import types

from mcp_client import call_tool, connect, discover_tools
from observability import emit, new_trace_id, span

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
MAX_ROUNDS = int(os.environ.get("WEATHER_AGENT_MAX_ROUNDS", "4"))
ALLOWED_TOOLS = {
    "get_weather",
    "get_forecast",
    "get_weather_alerts",
    "assess_weather_risk",
    "search_weather",
    "ask_weather",
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


_GEMINI_CLIENT: genai.Client | None = None


def _gemini_client() -> genai.Client:
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in the process environment")
        _GEMINI_CLIENT = genai.Client(api_key=api_key)
    return _GEMINI_CLIENT


def _gemini_models() -> list[str]:
    primary = MODEL
    configured = os.environ.get(
        "GEMINI_FALLBACK_MODELS",
        "gemini-3.5-flash-lite,gemini-2.5-flash-lite",
    )
    return list(dict.fromkeys([primary] + [m.strip() for m in configured.split(",") if m.strip()]))


def _retryable(exc: Exception) -> bool:
    text = str(exc).upper()
    return any(x in text for x in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL", "504", "DEADLINE_EXCEEDED"))


def _thinking_level() -> str:
    value = os.environ.get("GEMINI_THINKING_LEVEL", "low").strip().lower()
    return value if value in {"minimal", "low", "medium", "high"} else "low"


def _max_output_tokens() -> int:
    return max(128, int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "700")))


def _gemini_config(tool_declarations: list[types.FunctionDeclaration]) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0,
        max_output_tokens=_max_output_tokens(),
        thinking_config=types.ThinkingConfig(thinking_level=_thinking_level()),
        tools=[types.Tool(function_declarations=tool_declarations)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


async def _generate(contents: list[Any], config: types.GenerateContentConfig) -> tuple[Any, str]:
    """Call Gemini without blocking the MCP event loop, with transient fallback."""
    client = _gemini_client()
    errors: list[str] = []
    for model in _gemini_models():
        for attempt in range(2):
            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=contents,
                    config=config,
                )
                return response, model
            except Exception as exc:
                errors.append(f"{model} attempt {attempt + 1}: {exc}")
                if not _retryable(exc):
                    raise
                if attempt == 0:
                    await asyncio.sleep(1.0)
    raise RuntimeError("All configured Gemini models failed: " + " | ".join(errors))


def _is_simple_current_weather_query(query: str) -> bool:
    text = query.lower().strip()
    if any(x in text for x in ("tomorrow", "forecast", "next week", "this week")):
        return False
    patterns = (r"\bcurrent weather\b", r"\bcurrent conditions?\b", r"\bweather right now\b", r"\bweather now\b", r"\bwhat(?:'s| is) the weather\b", r"\bhow is the weather\b")
    return any(re.search(pattern, text) for pattern in patterns)


def _is_comparison_query(query: str) -> bool:
    text = query.lower().strip()
    return any(phrase in text for phrase in ("compare ", "compare the ", "which is better", "which would be better", "between ", "versus ", " vs "))


def _is_direct_rag_query(query: str) -> bool:
    text = query.lower().strip()
    if _is_simple_current_weather_query(query) or _is_comparison_query(query):
        return False
    if any(x in text for x in ("tomorrow", "today", "right now", "this evening", "tonight", "forecast", "next week")):
        return False
    return any(marker in text for marker in ("typically", "usually", "associated with", "what causes", "what conditions", "why does", "how does", "meaning of", "what is", "what are"))


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
    return (f"Forecast for {name} on {forecast.get('date')}: {forecast.get('condition')}, "
            f"{forecast.get('temperature_min_c')}–{forecast.get('temperature_max_c')}°C, "
            f"{forecast.get('precipitation_probability_pct')}% precipitation probability, "
            f"{forecast.get('precipitation_mm')} mm precipitation, and maximum wind "
            f"{forecast.get('max_wind_kmh')} km/h.")


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
    if not sources or re.search(r"(?im)^\s*sources\s*:?\s*$", answer):
        return answer
    lines = ["", "Sources:"]
    for source in sources:
        citation = source.get("citation", "")
        title = source.get("title") or source.get("source") or "Local weather knowledge base"
        topic = source.get("topic")
        suffix = f" — {topic}" if topic else ""
        lines.append(f"- [{citation}] {title}{suffix}")
    return answer.rstrip() + "\n" + "\n".join(lines)


def _gemini_tool_declarations(discovered: list[dict[str, Any]]) -> list[types.FunctionDeclaration]:
    declarations: list[types.FunctionDeclaration] = []
    for tool in discovered:
        function = tool["function"]
        declarations.append(
            types.FunctionDeclaration(
                name=function["name"],
                description=function.get("description", ""),
                parameters=function.get("parameters") or {"type": "object", "properties": {}},
            )
        )
    return declarations


async def run_agent(query: str) -> dict[str, Any]:
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty")

    state = AgentState(trace_id=new_trace_id(), query=query)
    started = time.perf_counter()
    emit("agent.start", trace_id=state.trace_id, query=query, model=MODEL, provider="gemini")
    simple_current = _is_simple_current_weather_query(query)
    comparison_query = _is_comparison_query(query)

    async with connect(trace_id=state.trace_id) as session:
        discovered = await discover_tools(session)
        tools = [t for t in discovered if t["function"]["name"] in ALLOWED_TOOLS]
        declarations = _gemini_tool_declarations(tools)
        config = _gemini_config(declarations)
        contents: list[Any] = [types.Content(role="user", parts=[types.Part.from_text(text=query)])]
        retrieval_failure = False

        if _is_direct_rag_query(query):
            with span("agent.route", trace_id=state.trace_id, route="direct_rag") as info:
                args = {"query": query}
                with span("tool.search_weather", trace_id=state.trace_id, tool="search_weather") as tool_info:
                    result = await call_tool(session, "search_weather", args)
                    success = bool(result.get("success", True)) if isinstance(result, dict) else True
                    tool_info["success"] = success
                state.tool_calls.append({"name": "search_weather", "arguments": args})
                state.observations.append({"tool": "search_weather", "result": result})
                emit("agent.tool", trace_id=state.trace_id, tool="search_weather", arguments=args, success=success, route="direct_rag")
                info["tool"] = "search_weather"
                info["success"] = success
                info["tool_calls"] = 1
            if not isinstance(result, dict) or not result.get("success"):
                answer = "I could not produce a grounded answer because the weather knowledge retrieval service failed. Please retry after the retrieval service is available."
                emit("agent.end", trace_id=state.trace_id, rounds=1, tools=1, route="direct_rag", success=False, latency_ms=round((time.perf_counter() - started) * 1000, 2))
                return {"success": False, "answer": answer, "trace_id": state.trace_id, "rounds": 1, "tool_calls": state.tool_calls, "observations": state.observations}
            answer = _append_rag_sources(str(result.get("answer") or "The retrieval tool returned no grounded answer."), state.observations)
            emit("agent.end", trace_id=state.trace_id, rounds=1, tools=1, route="direct_rag", success=True, latency_ms=round((time.perf_counter() - started) * 1000, 2))
            return {"success": True, "answer": answer, "trace_id": state.trace_id, "rounds": 1, "tool_calls": state.tool_calls, "observations": state.observations, "sources": _rag_sources(state.observations), "route": "direct_rag"}

        for round_no in range(1, MAX_ROUNDS + 1):
            with span("agent.reason", trace_id=state.trace_id, round=round_no) as info:
                response, used_model = await _generate(contents, config)
                info["tool_calls"] = len(response.function_calls or [])
                info["model"] = used_model

            candidate = response.candidates[0] if response.candidates else None
            if candidate is None or candidate.content is None:
                raise RuntimeError("Gemini returned no candidate content")

            function_calls = []
            for part in candidate.content.parts or []:
                if part.function_call:
                    function_calls.append(part.function_call)

            if not function_calls:
                answer = "I could not produce a grounded answer because the weather knowledge retrieval service failed. Please retry after the retrieval service is available." if retrieval_failure else _append_rag_sources((response.text or "").strip(), state.observations)
                emit("agent.end", trace_id=state.trace_id, rounds=round_no, tools=len(state.tool_calls), provider="gemini", model=used_model, success=not retrieval_failure, latency_ms=round((time.perf_counter() - started) * 1000, 2))
                return {"success": not retrieval_failure, "answer": answer, "trace_id": state.trace_id, "rounds": round_no, "tool_calls": state.tool_calls, "observations": state.observations, "sources": _rag_sources(state.observations)}

            contents.append(candidate.content)

            async def execute(function_call: Any) -> tuple[str, dict[str, Any], Any]:
                name = function_call.name
                args = dict(function_call.args or {})
                if name not in ALLOWED_TOOLS:
                    return name, args, {"success": False, "error": f"Tool '{name}' is not allowed."}
                try:
                    with span("tool." + name, trace_id=state.trace_id, tool=name) as tool_info:
                        result = await call_tool(session, name, args)
                        tool_info["success"] = bool(result.get("success", True)) if isinstance(result, dict) else True
                    return name, args, result
                except Exception as exc:
                    return name, args, {"success": False, "error": str(exc)}

            with span("agent.execute_tools", trace_id=state.trace_id, round=round_no) as info:
                results = await asyncio.gather(*(execute(fc) for fc in function_calls))
                info["executed"] = len(results)

            response_parts: list[types.Part] = []
            for function_call, (name, args, result) in zip(function_calls, results):
                state.tool_calls.append({"name": name, "arguments": args})
                state.observations.append({"tool": name, "result": result})
                success = result.get("success", True) if isinstance(result, dict) else True
                emit("agent.tool", trace_id=state.trace_id, tool=name, arguments=args, success=success, provider="gemini")
                if name in {"search_weather", "ask_weather"} and not success:
                    retrieval_failure = True

                response_parts.append(
                    types.Part.from_function_response(
                        name=name,
                        response=result if isinstance(result, dict) else {"result": result},
                        id=getattr(function_call, "id", None),
                    )
                )

                if simple_current and name == "get_weather" and isinstance(result, dict) and result.get("success"):
                    answer = _render_current_weather(result)
                    if answer:
                        emit("agent.end", trace_id=state.trace_id, rounds=round_no, tools=len(state.tool_calls), provider="gemini", deterministic=True, latency_ms=round((time.perf_counter() - started) * 1000, 2))
                        return {"success": True, "answer": answer, "trace_id": state.trace_id, "rounds": round_no, "tool_calls": state.tool_calls, "observations": state.observations, "deterministic": True}

                if (not comparison_query and name == "get_forecast" and isinstance(result, dict) and result.get("success") and any(x in query.lower() for x in ("tomorrow", "forecast", "weather be like"))):
                    answer = _render_forecast(result)
                    if answer:
                        emit("agent.end", trace_id=state.trace_id, rounds=round_no, tools=len(state.tool_calls), provider="gemini", deterministic=True, latency_ms=round((time.perf_counter() - started) * 1000, 2))
                        return {"success": True, "answer": answer, "trace_id": state.trace_id, "rounds": round_no, "tool_calls": state.tool_calls, "observations": state.observations, "deterministic": True}

            contents.append(types.Content(role="user", parts=response_parts))

        return {"success": False, "answer": "I could not gather enough evidence within the agent round limit.", "trace_id": state.trace_id, "rounds": MAX_ROUNDS, "tool_calls": state.tool_calls, "observations": state.observations}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Gemini MCP weather agent.")
    parser.add_argument("query", nargs="*")
    args = parser.parse_args()
    prompt = " ".join(args.query).strip() or input("Weather question: ").strip()
    result = asyncio.run(run_agent(prompt))
    print(result["answer"])
    print(f"\ntrace_id={result['trace_id']} rounds={result['rounds']}")


if __name__ == "__main__":
    main()
