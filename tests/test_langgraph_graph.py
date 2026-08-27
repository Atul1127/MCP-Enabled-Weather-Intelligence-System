"""Contract tests for the LangGraph orchestration layer."""
import asyncio

from weather_agent_core.graph import build_weather_graph


def test_graph_routes_tool_round_back_to_reasoner():
    async def run():
        seen=[]
        async def router(state): seen.append("router"); return {}
        async def planner(state): seen.append("planner"); return {}
        async def reasoner(state):
            seen.append("reasoner")
            if state.get("rounds", 0) == 0: return {"next_action":"tool", "rounds":1, "pending_calls":["tool"]}
            return {"next_action":"finish", "rounds":2}
        async def executor(state): seen.append("executor"); return {}
        async def synthesizer(state): seen.append("synthesizer"); return {"answer":"ok"}
        graph=build_weather_graph(router=router,planner=planner,reasoner=reasoner,executor=executor,synthesizer=synthesizer,max_rounds=2)
        result=await graph.ainvoke({"rounds":0})
        assert result["answer"] == "ok"
        assert seen == ["router","planner","reasoner","executor","reasoner","synthesizer"]
    asyncio.run(run())


def test_graph_respects_max_rounds():
    async def run():
        calls=[]
        async def router(state): calls.append("router"); return {}
        async def planner(state): calls.append("planner"); return {}
        async def reasoner(state): calls.append("reasoner"); return {"next_action":"tool","rounds":int(state.get("rounds",0))+1}
        async def executor(state): calls.append("executor"); return {}
        async def synthesizer(state): return {"answer":"bounded"}
        graph=build_weather_graph(router=router,planner=planner,reasoner=reasoner,executor=executor,synthesizer=synthesizer,max_rounds=1)
        result=await graph.ainvoke({"rounds":0})
        assert result["answer"] == "bounded"
        assert calls == ["router","planner","reasoner"]
    asyncio.run(run())
