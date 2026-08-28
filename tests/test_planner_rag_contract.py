from weather_agent_core.planner import Planner


def test_knowledge_plan_requires_rag_mcp_capability():
    plan = Planner().build("What does precipitation probability actually mean?")
    assert plan["intent"] == "knowledge"
    assert plan["requires_knowledge"] is True
    knowledge_steps = [
        step
        for step in plan["steps"]
        if step["capability"] == "knowledge"
    ]
    assert len(knowledge_steps) == 1
    assert knowledge_steps[0]["required"] is True
    assert set(knowledge_steps[0]["preferred_tools"]) == {
        "search_weather",
        "ask_weather",
    }


def test_weather_knowledge_benchmark_queries_route_to_knowledge():
    queries = (
        "Which WMO weather code represents a thunderstorm?",
        "Why does forecast uncertainty increase for some future weather claims?",
    )
    for query in queries:
        plan = Planner().build(query)
        assert plan["intent"] == "knowledge"
        assert plan["requires_knowledge"] is True
        knowledge_steps = [
            step for step in plan["steps"] if step["capability"] == "knowledge"
        ]
        assert len(knowledge_steps) == 1
        assert knowledge_steps[0]["required"] is True
