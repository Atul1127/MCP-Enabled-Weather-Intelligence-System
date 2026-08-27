"""Orchestrator: Router -> Planner -> Gemini tool selection -> MCP Executor -> Synthesizer."""
from __future__ import annotations

import asyncio
import os
from typing import Any

from google import genai
from google.genai import types

from mcp_client import connect, discover_tools
from observability import emit, new_trace_id, span

from .executor import MCPExecutor
from .planner import Planner
from .state import AgentState
from .synthesizer import GeminiSynthesizer


ALLOWED_TOOLS = {
    "get_weather", "get_forecast", "get_weather_alerts",
    "assess_weather_risk", "search_weather", "ask_weather",
}
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
MAX_ROUNDS = max(1, int(os.environ.get("WEATHER_AGENT_MAX_ROUNDS", "4")))


class WeatherAgent:
    """Stateful orchestration boundary for the weather intelligence system."""

    def __init__(self, model: str = DEFAULT_MODEL, max_rounds: int = MAX_ROUNDS):
        self.model = model
        self.max_rounds = max_rounds
        self.planner = Planner()
        self._client: genai.Client | None = None

    def _client_or_raise(self) -> genai.Client:
        if self._client is None:
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY is not set in the process environment")
            self._client = genai.Client(api_key=key)
        return self._client

    @staticmethod
    def _declarations(discovered: list[dict[str, Any]]) -> list[types.FunctionDeclaration]:
        declarations = []
        for tool in discovered:
            fn = tool["function"]
            if fn["name"] not in ALLOWED_TOOLS:
                continue
            declarations.append(types.FunctionDeclaration(
                name=fn["name"],
                description=fn.get("description", ""),
                parameters=fn.get("parameters") or {"type": "object", "properties": {}},
            ))
        return declarations

    async def _reason(self, contents: list[Any], declarations: list[types.FunctionDeclaration], plan: dict) -> Any:
        prompt = (
            "You are the planning/reasoning layer of an Indian Weather Intelligence agent.\n"
            "Follow the execution plan, but use MCP tools as the source of truth.\n"
            "Never invent live weather data. For comparisons, collect evidence for every requested location.\n"
            "Use search_weather/ask_weather for conceptual or historical knowledge.\n\n"
            f"Execution plan: {plan}"
        )
        config = types.GenerateContentConfig(
            system_instruction=prompt,
            temperature=0,
            max_output_tokens=700,
            tools=[types.Tool(function_declarations=declarations)],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        return await asyncio.to_thread(
            self._client_or_raise().models.generate_content,
            model=self.model,
            contents=contents,
            config=config,
        )

    async def run(self, query: str) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty")

        trace_id = new_trace_id()
        state = AgentState(query=query, trace_id=trace_id)
        plan = self.planner.build(query)
        state.intent = plan["intent"]
        state.plan = plan["steps"]
        state.route = "knowledge" if state.intent == "knowledge" else "mcp_agent"
        emit("agent.start", trace_id=trace_id, intent=state.intent, route=state.route, model=self.model)

        async with connect(trace_id=trace_id) as session:
            discovered = await discover_tools(session)
            declarations = self._declarations(discovered)
            executor = MCPExecutor(session, ALLOWED_TOOLS)
            contents: list[Any] = [types.Content(role="user", parts=[types.Part.from_text(text=query)])]

            for round_no in range(1, self.max_rounds + 1):
                with span("agent.reason", trace_id=trace_id, round=round_no) as info:
                    response = await self._reason(contents, declarations, plan)
                    calls = list(response.function_calls or [])
                    info["tool_calls"] = len(calls)
                    info["model"] = self.model

                candidate = response.candidates[0] if response.candidates else None
                if candidate is None or candidate.content is None:
                    raise RuntimeError("Gemini returned no candidate content")

                if not calls:
                    # No new evidence was requested. Let the dedicated synthesizer
                    # produce the final answer from the state accumulated so far.
                    break

                contents.append(candidate.content)
                with span("agent.execute", trace_id=trace_id, round=round_no) as info:
                    results = await executor.execute(calls)
                    info["executed"] = len(results)

                response_parts: list[types.Part] = []
                for function_call, (name, args, result) in zip(calls, results):
                    state.add_observation(name, args, result)
                    if isinstance(result, dict) and not result.get("success", True):
                        state.errors.append(f"{name}: {result.get('error', 'tool failed')}")
                    emit("agent.tool", trace_id=trace_id, tool=name, success=not state.errors[-1:] or not state.errors[-1].startswith(name + ":"))
                    response_parts.append(types.Part.from_function_response(
                        name=name,
                        response=result if isinstance(result, dict) else {"result": result},
                        id=getattr(function_call, "id", None),
                    ))
                    if isinstance(result, dict):
                        for source in result.get("sources") or []:
                            if isinstance(source, dict) and source not in state.sources:
                                state.sources.append(source)
                contents.append(types.Content(role="user", parts=response_parts))

        with span("agent.synthesize", trace_id=trace_id) as info:
            answer = await GeminiSynthesizer(self._client_or_raise(), self.model).synthesize(query, state)
            info["answer_chars"] = len(answer)

        success = not state.retrieval_failed and not any(
            isinstance(obs.get("result"), dict) and obs["result"].get("success") is False
            for obs in state.observations
            if obs["tool"] in {"get_weather", "get_forecast", "get_weather_alerts", "assess_weather_risk"}
        )
        emit("agent.end", trace_id=trace_id, intent=state.intent, rounds=self.max_rounds, tools=len(state.tool_calls), success=success)
        return {
            "success": success,
            "answer": answer,
            "trace_id": trace_id,
            "intent": state.intent,
            "route": state.route,
            "plan": state.plan,
            "tool_calls": state.tool_calls,
            "observations": state.observations,
            "sources": state.sources,
        }
