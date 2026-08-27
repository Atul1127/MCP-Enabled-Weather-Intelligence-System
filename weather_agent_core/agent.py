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
from .executor import MCPExecutor
from .graph import build_weather_graph
from .planner import Planner
from .router import classify
from .state import AgentState
from .synthesizer import GeminiSynthesizer
from .graph.state import GraphState

ALLOWED_TOOLS = {"get_weather", "get_forecast", "get_weather_alerts", "assess_weather_risk", "search_weather", "ask_weather"}
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
MAX_ROUNDS = max(1, int(os.environ.get("WEATHER_AGENT_MAX_ROUNDS", "4")))

class WeatherAgent:
    """Application boundary; LangGraph owns orchestration and MCP owns capabilities."""
    def __init__(self, model: str = DEFAULT_MODEL, max_rounds: int = MAX_ROUNDS):
        if max_rounds < 1: raise ValueError("max_rounds must be at least 1")
        self.model, self.max_rounds = model, max_rounds
        self.planner = Planner(); self._client: genai.Client | None = None

    def _client_or_raise(self) -> genai.Client:
        if self._client is None:
            key = os.environ.get("GEMINI_API_KEY")
            if not key: raise RuntimeError("GEMINI_API_KEY is not set in the process environment")
            self._client = genai.Client(api_key=key)
        return self._client

    @staticmethod
    def _declarations(discovered: list[dict[str, Any]]) -> list[types.FunctionDeclaration]:
        declarations=[]
        for tool in discovered:
            function=tool.get("function", {}); name=function.get("name")
            if name in ALLOWED_TOOLS:
                declarations.append(types.FunctionDeclaration(name=name, description=function.get("description", ""), parameters=function.get("parameters") or {"type":"object","properties":{}}))
        return declarations

    async def _reason(self, messages: list[types.Content], declarations: list[types.FunctionDeclaration], plan: dict[str, Any]):
        kwargs: dict[str, Any]={"system_instruction":"You are the execution-selection layer. Follow the explicit plan. Use MCP tools for live evidence and weather knowledge. Complete every required plan step before stopping. Gather every required location for comparisons. Never invent live values.\n\nPLAN:\n"+str(plan),"max_output_tokens":700,"tools":[types.Tool(function_declarations=declarations)],"automatic_function_calling":types.AutomaticFunctionCallingConfig(disable=True)}
        if not self.model.startswith(("gemini-3.5","gemini-3.6","gemini-3.7")): kwargs["temperature"]=0
        return await asyncio.to_thread(self._client_or_raise().models.generate_content,model=self.model,contents=messages,config=types.GenerateContentConfig(**kwargs))

    async def run(self, query: str) -> dict[str, Any]:
        query=query.strip()
        if not query: raise ValueError("Query cannot be empty")
        trace_id=new_trace_id(); runtime=AgentState(query=query,trace_id=trace_id)
        emit("agent.start",trace_id=trace_id,model=self.model)
        async with connect(trace_id=trace_id) as session:
            declarations=self._declarations(await discover_tools(session))
            if not declarations: raise RuntimeError("MCP server exposed no allowed tools")
            executor=MCPExecutor(session,ALLOWED_TOOLS); messages=[types.Content(role="user",parts=[types.Part.from_text(text=query)])]

            async def router_node(_: GraphState) -> dict[str,Any]:
                # Routing is deterministic and remains separate from planning.
                return {"intent": classify(query)}

            async def planner_node(state: GraphState) -> dict[str,Any]:
                plan=self.planner.build(query)
                runtime.intent=plan["intent"]; runtime.plan=plan
                runtime.required_tool_groups=[set(step["preferred_tools"]) for step in plan["steps"] if step.get("required",True) or (plan["requires_knowledge"] and step.get("capability")=="knowledge")]
                runtime.route="rag" if plan["requires_knowledge"] and not plan["requires_live_data"] else "mcp+rag" if plan["requires_knowledge"] and plan["requires_live_data"] else "mcp"
                if state.get("intent") and state["intent"] != runtime.intent: raise RuntimeError("Router and planner intent disagree")
                return {"plan":plan}

            async def reasoner_node(state: GraphState) -> dict[str,Any]:
                round_no=int(state.get("rounds",0))+1
                with span("agent.reason",trace_id=trace_id,round=round_no) as info:
                    response=await self._reason(messages,declarations,runtime.plan)
                    calls=list(response.function_calls or []); info.update(tool_calls=len(calls),model=self.model)
                candidate=response.candidates[0] if response.candidates else None
                if candidate is None or candidate.content is None: raise RuntimeError("Gemini returned no candidate content")
                if not calls: return {"next_action":"finish","rounds":round_no}
                messages.append(candidate.content)
                return {"next_action":"tool","pending_calls":calls,"rounds":round_no}

            async def executor_node(state: GraphState) -> dict[str,Any]:
                calls=list(state.get("pending_calls",[])); results=await executor.execute(calls); response_parts=[]
                for function_call,(name,args,result) in zip(calls,results):
                    runtime.add_observation(name,args,result)
                    emit("agent.tool",trace_id=trace_id,tool=name,success=not(isinstance(result,dict) and result.get("success") is False),round=int(state.get("rounds",0)))
                    response_parts.append(types.Part.from_function_response(name=name,response=result if isinstance(result,dict) else {"result":result},id=getattr(function_call,"id",None)))
                messages.append(types.Content(role="user",parts=response_parts))
                return {"observations":runtime.observations,"tool_calls":runtime.tool_calls,"evidence":runtime.evidence_payload(),"sources":runtime.sources,"errors":runtime.errors}

            async def synthesizer_node(_: GraphState) -> dict[str,Any]:
                answer=await GeminiSynthesizer(self._client_or_raise(),self.model).synthesize(query,runtime)
                answer,cited_sources=validate_citations(answer,runtime.sources)
                if cited_sources: runtime.sources=cited_sources
                return {"answer":answer}

            graph=build_weather_graph(router=router_node,planner=planner_node,reasoner=reasoner_node,executor=executor_node,synthesizer=synthesizer_node,max_rounds=self.max_rounds)
            result=await graph.ainvoke({"query":query,"trace_id":trace_id,"rounds":0})
        success=runtime.required_requirements_satisfied
        emit("agent.end",trace_id=trace_id,intent=runtime.intent,rounds=result.get("rounds",0),tools=len(runtime.tool_calls),success=success)
        return {"success":success,"answer":result.get("answer",""),"trace_id":trace_id,"intent":runtime.intent,"route":runtime.route,"plan":runtime.plan,"tool_calls":runtime.tool_calls,"observations":runtime.observations,"evidence":runtime.evidence_payload(),"sources":runtime.sources,"errors":runtime.errors,"rounds":result.get("rounds",0)}
