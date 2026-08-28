from weather_agent_core.planner import Planner


def test_knowledge_plan_requires_both_rag_mcp_capabilities():
    plan = Planner().build("What does precipitation probability actually mean?")
    assert plan["intent"] == "knowledge"
    assert plan["requires_knowledge"] is True

    required_steps = [
        step for step in plan["steps"] if step.get("required", True)
    ]
    assert [step["preferred_tools"] for step in required_steps] == [
        ("search_weather",),
        ("ask_weather",),
    ]
    assert all(step["required"] is True for step in required_steps)
    assert all(step["parallelizable"] is False for step in required_steps)


def test_weather_knowledge_benchmark_queries_route_to_knowledge():
    queries = (
        "Which WMO weather code represents a thunderstorm?",
        "Why does forecast uncertainty increase for some future weather claims?",
    )
    for query in queries:
        plan = Planner().build(query)
        assert plan["intent"] == "knowledge"
        assert plan["requires_knowledge"] is True
        required_tools = [
            step["preferred_tools"][0]
            for step in plan["steps"]
            if step.get("required", True)
        ]
        assert required_tools == ["search_weather", "ask_weather"]
