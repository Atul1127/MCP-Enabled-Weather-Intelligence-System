"""Contract tests for the LangGraph orchestration layer."""
import pytest

from weather_agent_core.graph import build_weather_graph


@pytest.mark.asyncio
async def test_graph_routes_tool_round_back_to_reasoner():
    seen=[]

    async def router(state):
        seen.append("router"); return {}
    async def planner(state):
        seen.append("planner"); return {}
    async def reasoner(state):
        seen.append("reasoner")
        if state.get("rounds", 0) == 0:
            return {"next_action": "tool", "rounds": 1, "pending_calls": ["tool"]}
        return {"next_action": "finish", "rounds": 2}
    async def executor(state):
        seen.append("executor"); return {}
    async def synthesizer(state):
        seen.append("synthesizer"); return {"answer": "ok"}

    graph=build_weather_graph(router=router, planner=planner, reasoner=reasoner, executor=executor, synthesizer=synthesizer, max_rounds=2)
    result=await graph.ainvoke({"rounds": 0})
    assert result["answer"] == "ok"
    assert seen == ["router", "planner", "reasoner", "executor", "reasoner", "synthesizer"]


@pytest.mark.asyncio
async def test_graph_respects_max_rounds():
    calls=[]
    async def node(name, state): calls.append(name); return {"next_action":"tool","rounds":int(state.get("rounds",0))+1} if name=="reasoner" else {}
    async def router(s): return await node("router",s)
    async def planner(s): return await node("planner",s)
    async def reasoner(s): return await node("reasoner",s)
    async def executor(s): return await node("executor",s)
    async def synthesizer(s): return {"answer":"bounded"}
    graph=build_weather_graph(router=router, planner=planner, reasoner=reasoner, executor=executor, synthesizer=synthesizer, max_rounds=1)
    result=await graph.ainvoke({"rounds":0})
    assert result["answer"] == "bounded"
    assert calls == ["router", "planner", "reasoner"]
