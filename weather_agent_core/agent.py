"""Stateful Gemini agent: Router -> Planner -> Reasoner -> MCP Executor -> Synthesizer."""
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
    """Application orchestration boundary; capabilities remain behind MCP."""

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
        return [
            types.FunctionDeclaration(
                name=tool["function"]["name"],
                description=tool["function"].get("description", ""),
                parameters=tool["function"].get("parameters") or {"type": "object", "properties": {}},
            )
            for tool in discovered
            if tool["function"]["name"] in ALLOWED_TOOLS
        ]

    async def _reason(self, contents: list[Any], declarations: list[types.FunctionDeclaration], plan: dict[str, Any]) -> Any:
        instruction = (
            "You are the reasoning/execution-selection layer of an Indian Weather Intelligence agent. "
            "Follow the explicit plan and use MCP tools for evidence. Never invent live values. "
            "For comparison requests, gather evidence for every requested location. "
            "Use knowledge tools for conceptual guidance. Do not answer until required evidence has been collected.\n\n"
            f"Plan:\n{plan}"
        )
        config = types.GenerateContentConfig(
            system_instruction=instruction,
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
        state.plan = plan
        state.route = "rag" if state.intent == "knowledge" else "mcp+rAG" if plan.get("requires_knowledge") else "mcp"
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
                    info.update(tool_calls=len(calls), model=self.model)

                candidate = response.candidates[0] if response.candidates else None
                if candidate is None or candidate.content is None:
                    raise RuntimeError("Gemini returned no candidate content")
                if not calls:
                    break

                contents.append(candidate.content)
                with span("agent.execute", trace_id=trace_id, round=round_no) as info:
                    results = await executor.execute(calls)
                    info["executed"] = len(results)

                response_parts: list[types.Part] = []
                for function_call, (name, args, result) in zip(calls, results):
                    state.add_observation(name, args, result)
                    success = not (isinstance(result, dict) and result.get("success") is False)
                    emit("agent.tool", trace_id=trace_id, tool=name, success=success, round=round_no)
                    response_parts.append(types.Part.from_function_response(
                        name=name,
                        response=result if isinstance(result, dict) else {"result": result},
                        id=getattr(function_call, "id", None),
                    ))
                contents.append(types.Content(role="user", parts=response_parts))

        with span("agent.synthesize", trace_id=trace_id) as info:
            answer = await GeminiSynthesizer(self._client_or_raise(), self.model).synthesize(query, state)
            info["answer_chars"] = len(answer)

        success = not state.has_live_failure and not state.retrieval_failed
        emit("agent.end", trace_id=trace_id, intent=state.intent, rounds=min(self.max_rounds, len(state.observations) + 1), tools=len(state.tool_calls), success=success)
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
            "errors": state.errors,
        }
