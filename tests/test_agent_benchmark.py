from agent import _is_comparison_query
from evaluation.agent_benchmark import arg_match, evaluate_case


def test_arg_match_normalizes_scalar_arguments():
    assert arg_match(
        {"location": "Kolkata", "date": "tomorrow"},
        {"tool": "get_forecast", "location": "kolkata", "date": "tomorrow"},
    )


def test_arg_match_allows_activity_phrase_overlap():
    assert arg_match(
        {"location": "Mumbai", "activity": "outdoor running", "date": "tomorrow"},
        {"tool": "assess_weather_risk", "location": "Mumbai", "activity": "outdoor run", "date": "tomorrow"},
    )


def test_evaluate_case_detects_required_multi_tool_calls():
    case = {
        "id": "compare",
        "category": "multi_tool",
        "question": "Compare Kolkata and Mumbai tomorrow",
        "expected_tools": ["assess_weather_risk"],
        "required_args": [
            {"tool": "assess_weather_risk", "location": "Kolkata", "date": "tomorrow"},
            {"tool": "assess_weather_risk", "location": "Mumbai", "date": "tomorrow"},
        ],
    }
    result = {
        "success": True,
        "tool_calls": [
            {"name": "assess_weather_risk", "arguments": {"location": "Kolkata", "date": "tomorrow"}},
            {"name": "assess_weather_risk", "arguments": {"location": "Mumbai", "date": "tomorrow"}},
        ],
        "rounds": 2,
        "trace_id": "test",
    }
    evaluated = evaluate_case(case, result, 10.0)
    assert evaluated["tool_selection_correct"]
    assert evaluated["argument_accuracy"] == 1
    assert evaluated["unnecessary_tool_calls"] == 0


def test_comparison_query_disables_single_forecast_shortcut():
    assert _is_comparison_query("Compare the weather forecast for Kolkata and Delhi tomorrow.")
    assert _is_comparison_query("Which is better for cricket tomorrow, Kolkata or Mumbai?")
    assert not _is_comparison_query("What will the weather be like in Kolkata tomorrow?")
