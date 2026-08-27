from evaluation.agent_e2e_eval import evaluate_case


def test_e2e_case_measures_tool_and_argument_accuracy():
    case = {
        "id": "x",
        "category": "current",
        "expected_tools": ["get_weather"],
        "required_args": [{"tool": "get_weather", "location": "Kolkata"}],
    }
    result = {
        "success": True,
        "answer": "Kolkata is warm.",
        "observations": [{"tool": "get_weather", "result": {"success": True}}],
        "tool_calls": [{"name": "get_weather", "arguments": {"location": "Kolkata"}}],
        "verification": {"sufficient": True},
        "citations": [],
        "errors": [],
        "rounds": 1,
        "retry_count": 0,
    }
    metrics = evaluate_case(case, result, 12.5)
    assert metrics["success"] is True
    assert metrics["tool_selection_recall"] == 1.0
    assert metrics["argument_accuracy"] == 1.0
    assert metrics["evidence_sufficient"] is True


def test_e2e_case_does_not_count_failed_tool_as_successful():
    case = {"id": "x", "expected_tools": ["get_weather"], "required_args": []}
    result = {
        "success": False,
        "answer": "",
        "observations": [{"tool": "get_weather", "result": {"success": False}}],
        "tool_calls": [],
        "verification": {"sufficient": False},
        "citations": [],
        "errors": ["get_weather: failed"],
        "rounds": 1,
        "retry_count": 1,
    }
    metrics = evaluate_case(case, result, 4.0)
    assert metrics["success"] is False
    assert metrics["tool_selection_recall"] == 0.0
    assert metrics["evidence_sufficient"] is False
    assert metrics["retries"] == 1
