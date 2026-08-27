"""LangGraph-orchestrated Gemini agent with MCP capabilities and unified evidence."""
from __future__ import annotations

import asyncio
import os
from typing import Any

from google import genai
from google.genai import types
from mcp_client import connect, discover_tools
from observability import emit, new_trace_id, span
from rag.citations.validator import validate as validate_citations

from .decomposer import decompose
from .executor import MCPExecutor
from .graph import build_weather_graph
from .graph.state import GraphState
from .mcp_registry import MCPCapabilityRegistry
from .mcp_router import MCPCapabilityRouter
from .planner import Planner
from .router import classify
from .security import inspect_text
from .state import AgentState
from .synthesizer import GeminiSynthesizer
from .verifier import EvidenceVerifier

ALLOWED_TOOLS = {
    "get_weather",
    "get_forecast",
    "get_weather_alerts",
    "assess_weather_risk",
    "search_weather",
    "ask_weather",
}
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
MAX_ROUNDS = max(1, int(os.environ.get("WEATHER_AGENT_MAX_ROUNDS", "4")))
MAX_RETRIES = max(0, int(os.environ.get("WEATHER_AGENT_MAX_RETRIES", "1")))


class WeatherAgent:
    """Application boundary; LangGraph owns orchestration and MCP owns capabilities."""

    def __init__(self, model: str = DEFAULT_MODEL, max_rounds: int = MAX_ROUNDS, max_retries: int = MAX_RETRIES):
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.model = model
        self.max_rounds = max_rounds
        self.max_retries = max_retries
        self.planner = Planner()
        self.verifier = EvidenceVerifier()
        self._client: genai.Client | None = None

    def _client_or_raise(self) -> genai.Client:
        if self._client is None:
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY is not set in the process environment")
            self._client = genai.Client(api_key=key)
        return self._client

    @staticmethod
    def _declarations(registry: MCPCapabilityRegistry, names: set[str] | None = None) -> list[types.FunctionDeclaration]:
        selected = registry.tools if names is None else tuple(item for item in registry.tools if item.name in names)
        return [types.FunctionDeclaration(name=item.name, description=item.description, parameters=item.schema) for item in selected]

    async def _reason(self, messages, declarations, plan, retry_reason=None):
        instruction = (
            "You are the execution-selection layer. Follow the explicit plan and its execution groups. "
            "Use only the MCP capabilities exposed in the current tool declarations. Use MCP tools for "
            "live evidence and weather knowledge. Complete every required plan step before stopping. "
            "Gather every required location for comparisons. Never invent live values.\n\nPLAN:\n" + str(plan)
        )
        if retry_reason:
            instruction += "\n\nVERIFIER FEEDBACK:\n" + retry_reason + "\nCorrect the missing evidence by selecting appropriate MCP tools from the exposed declarations."
        kwargs: dict[str, Any] = {
            "system_instruction": instruction,
            "max_output_tokens": 700,
            "tools": [types.Tool(function_declarations=declarations)],
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if not self.model.startswith(("gemini-3.5", "gemini-3.6", "gemini-3.7")):
            kwargs["temperature"] = 0
        return await asyncio.to_thread(self._client_or_raise().models.generate_content, model=self.model, contents=messages, config=types.GenerateContentConfig(**kwargs))

    async def run(self, query: str) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty")
        security = inspect_text(query)
        if security["suspicious"]:
            raise ValueError("Query contains a blocked prompt-injection signal")

        trace_id = new_trace_id()
        runtime = AgentState(query=query, trace_id=trace_id)
        emit("agent.start", trace_id=trace_id, model=self.model)

        async with connect(trace_id=trace_id) as session:
            discovered = await discover_tools(session)
            registry = MCPCapabilityRegistry(discovered, allowed_tools=ALLOWED_TOOLS)
            capability_router = MCPCapabilityRouter(registry)
            declarations = self._declarations(registry)
            if not declarations:
                raise RuntimeError("MCP server exposed no allowed tools")
            emit("agent.mcp_capabilities", trace_id=trace_id, **registry.summary())

            executor = MCPExecutor(session, registry.allowed_names)
            messages = [types.Content(role="user", parts=[types.Part.from_text(text=query)])]
            active_declarations = declarations
            structured_response: dict[str, Any] = {}

            async def router_node(_: GraphState) -> dict[str, Any]:
                return {"intent": classify(query), "route": "pending"}

            async def planner_node(state: GraphState) -> dict[str, Any]:
                nonlocal active_declarations
                plan = decompose(self.planner.build(query))
                runtime.intent = plan["intent"]
                runtime.plan = plan
                runtime.required_tool_groups = [set(step["preferred_tools"]) for step in plan["steps"] if step.get("required", True)]
                runtime.route = "mcp+rag" if plan["requires_knowledge"] and plan["requires_live_data"] else "rag" if plan["requires_knowledge"] else "mcp"
                if state.get("intent") and state["intent"] != runtime.intent:
                    raise RuntimeError("Router and planner intent disagree")
                route = capability_router.route_plan(plan)
                missing_groups = [
                    sorted(set(step.get("preferred_tools", [])) - set(route.selected))
                    for step in plan.get("steps", [])
                    if step.get("required", True) and not set(step.get("preferred_tools", [])) & set(route.selected)
                ]
                missing_groups = [group for group in missing_groups if group]
                if missing_groups:
                    missing = ", ".join("/".join(group) for group in missing_groups)
                    raise RuntimeError(f"MCP plan requires at least one available tool from: {missing}")
                active_declarations = self._declarations(registry, set(route.selected))
                emit("agent.mcp_route", trace_id=trace_id, requested=list(route.requested), selected=list(route.selected), rejected=list(route.rejected), execution_groups=plan.get("execution_groups", []))
                return {"plan": plan, "route": runtime.route, "mcp_tools": list(route.selected)}

            async def reasoner_node(state: GraphState) -> dict[str, Any]:
                round_no = int(state.get("rounds", 0)) + 1
                with span("agent.reason", trace_id=trace_id, round=round_no) as info:
                    response = await self._reason(messages, active_declarations, runtime.plan, state.get("retry_reason"))
                    calls = list(response.function_calls or [])
                    info.update(tool_calls=len(calls), model=self.model)
                candidate = response.candidates[0] if response.candidates else None
                if candidate is None or candidate.content is None:
                    raise RuntimeError("Gemini returned no candidate content")
                if not calls:
                    return {"next_action": "finish", "rounds": round_no, "candidate": candidate.content}
                messages.append(candidate.content)
                return {"next_action": "tool", "pending_calls": calls, "rounds": round_no, "retry_reason": None}

            async def executor_node(state: GraphState) -> dict[str, Any]:
                calls = list(state.get("pending_calls", []))
                results = await executor.execute(calls)
                response_parts = []
                for function_call, (name, args, result) in zip(calls, results):
                    runtime.add_observation(name, args, result)
                    emit("agent.tool", trace_id=trace_id, tool=name, success=not (isinstance(result, dict) and result.get("success") is False), round=int(state.get("rounds", 0)))
                    response_parts.append(types.Part.from_function_response(name=name, response=result if isinstance(result, dict) else {"result": result}, id=getattr(function_call, "id", None)))
                messages.append(types.Content(role="user", parts=response_parts))
                return {"observations": runtime.observations, "tool_calls": runtime.tool_calls, "evidence": runtime.evidence_payload(), "sources": runtime.sources, "errors": runtime.errors}

            async def verifier_node(state: GraphState) -> dict[str, Any]:
                verification = self.verifier.verify(runtime.plan, runtime.observations, runtime.evidence_payload(), runtime.errors)
                retry_count = int(state.get("retry_count", 0)) + (0 if verification["sufficient"] else 1)
                missing = ", ".join("/".join(group) for group in verification.get("missing_capabilities", []))
                return {"verification": verification, "retry_count": retry_count, "retry_reason": None if verification["sufficient"] else f"Missing required evidence capabilities: {missing}"}

            async def synthesizer_node(_: GraphState) -> dict[str, Any]:
                nonlocal structured_response
                structured_response = await GeminiSynthesizer(self._client_or_raise(), self.model).synthesize_structured(query, runtime)
                answer, cited_sources = validate_citations(structured_response["answer"], runtime.sources)
                structured_response["answer"] = answer
                structured_response["citations"] = [str(source["citation"]) for source in cited_sources if source.get("citation")]
                if cited_sources:
                    runtime.sources = cited_sources
                return {"answer": answer}

            graph = build_weather_graph(router=router_node, planner=planner_node, reasoner=reasoner_node, executor=executor_node, verifier=verifier_node, synthesizer=synthesizer_node, max_rounds=self.max_rounds, max_retries=self.max_retries)
            result = await graph.ainvoke({"query": query, "trace_id": trace_id, "rounds": 0, "retry_count": 0})

        success = runtime.required_requirements_satisfied
        emit("agent.end", trace_id=trace_id, intent=runtime.intent, rounds=result.get("rounds", 0), tools=len(runtime.tool_calls), success=success)
        return {"success": success, "answer": result.get("answer", ""), "confidence": structured_response.get("confidence"), "citations": structured_response.get("citations", []), "warnings": structured_response.get("warnings", []), "trace_id": trace_id, "intent": runtime.intent, "route": runtime.route, "plan": runtime.plan, "tool_calls": runtime.tool_calls, "observations": runtime.observations, "evidence": runtime.evidence_payload(), "sources": runtime.sources, "errors": runtime.errors, "rounds": result.get("rounds", 0), "retry_count": result.get("retry_count", 0), "verification": result.get("verification", {})}
