from weather_agent_core.decomposer import decompose


def test_decompose_groups_parallel_steps():
    plan = {"steps": [
        {"id": "weather", "parallelizable": True},
        {"id": "alerts", "parallelizable": True},
        {"id": "final", "parallelizable": False},
    ]}
    result = decompose(plan)
    assert result["execution_groups"] == [["weather", "alerts"], ["final"]]
    assert result["parallel_groups"] == [["weather", "alerts"]]
    assert result["task_count"] == 3


def test_decompose_preserves_plan_fields():
    plan = {"intent": "live_weather", "steps": [{"id": "weather"}], "requires_live_data": True}
    result = decompose(plan)
    assert result["intent"] == "live_weather"
    assert result["requires_live_data"] is True
