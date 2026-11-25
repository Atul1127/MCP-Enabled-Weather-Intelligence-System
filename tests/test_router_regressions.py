"""Regression tests for questions the keyword router used to misroute."""
import pytest

from weather_agent_core.agent import WeatherAgent
from weather_agent_core.planner import Planner
from weather_agent_core.router import classify, extract_location


@pytest.mark.parametrize(
    "query, expected",
    [
        # Place + time must use live data, even with conceptual words.
        ("What is the risk of rain in Mumbai tomorrow?", "live_weather"),
        ("What is the chance of rain in Chennai today?", "live_weather"),
        ("What is the visibility in Delhi today?", "live_weather"),
        ("Is a heatwave expected in Delhi this week?", "live_weather"),
        ("Will there be fog in Lucknow on Saturday?", "live_weather"),
        # "between 4 and 6 pm" is a time range, not a comparison.
        ("Will it rain between 4 and 6 pm in Pune?", "live_weather"),
        ("Will it rain between morning and evening in Pune today?", "live_weather"),
        # Place + measurable condition, no time word.
        ("What is the visibility in Delhi?", "live_weather"),
        ("Is it raining in Kolkata?", "live_weather"),
        # Place + time + advice/explanation needs live data AND knowledge.
        ("What precautions should I take in Mumbai tomorrow?", "live_knowledge"),
        ("Why is it so hot in Delhi today?", "live_knowledge"),
        # Activity questions stay activity risk.
        ("Should I run outdoors in Mumbai tomorrow?", "activity_risk"),
        ("Is it safe to travel to Shimla this weekend?", "activity_risk"),
        # Real comparisons still compare.
        ("Compare Delhi and Mumbai weather tomorrow", "comparison"),
        ("Which is warmer between Delhi and Jaipur today?", "comparison"),
        # Pure concepts stay in the knowledge base.
        ("What causes heavy rain in Mumbai?", "knowledge"),
        ("What precautions are useful during extreme heat?", "knowledge"),
        ("What are the typical risks of heavy rainfall?", "knowledge"),
        ("What does precipitation probability actually mean?", "knowledge"),
    ],
)
def test_routing(query, expected):
    assert classify(query) == expected


@pytest.mark.parametrize(
    "query, expected",
    [
        ("Weather in Mumbai in the evening", "Mumbai"),
        ("What is the risk of rain in Mumbai tomorrow?", "Mumbai"),
        ("Will it rain between 4 and 6 pm in Pune?", "Pune"),
        ("Forecast for New Delhi tomorrow", "New Delhi"),
        ("weather in bengaluru today", "bengaluru"),
        ("Is it safe to run at Marine Drive this evening?", "Marine Drive"),
        ("What precautions help during extreme heat?", None),
        ("What are the typical risks of heavy rainfall?", None),
    ],
)
def test_extract_location(query, expected):
    assert extract_location(query) == expected


def test_live_knowledge_plan_requires_live_data_and_adds_knowledge():
    plan = Planner().build("What precautions should I take in Mumbai tomorrow?")
    assert plan["intent"] == "live_knowledge"
    assert plan["requires_live_data"] is True
    assert plan["requires_knowledge"] is True
    required = [step for step in plan["steps"] if step["required"]]
    assert [step["capability"] for step in required] == ["live_weather"]
    assert required[0]["preferred_tools"][0] == "get_forecast"


def test_fallback_uses_extracted_location_and_records_added_calls():
    plan = Planner().build("What is the risk of rain in Mumbai in the evening tomorrow?")
    added: list[dict] = []
    calls = WeatherAgent._ensure_required_tool_calls([], plan, "What is the risk of rain in Mumbai in the evening tomorrow?", added=added)
    assert len(calls) == 1
    assert calls[0].args["location"] == "Mumbai"
    assert calls[0].args["date"] == "tomorrow"
    assert added == [{"name": calls[0].name, "arguments": calls[0].args}]


def test_fallback_skips_location_tools_when_no_place_is_found():
    plan = {"intent": "live_weather", "steps": [{"preferred_tools": ("get_weather",), "required": True}]}
    added: list[dict] = []
    assert WeatherAgent._ensure_required_tool_calls([], plan, "Will it rain tomorrow?", added=added) == []
    assert added == []


def test_explicit_date_from_gemini_is_kept():
    from google.genai import types

    plan = Planner().build("Compare Delhi today and Mumbai tomorrow")
    calls = [
        types.FunctionCall(name="get_forecast", args={"location": "Delhi", "date": "today"}),
        types.FunctionCall(name="get_forecast", args={"location": "Mumbai", "date": "tomorrow"}),
    ]
    result = WeatherAgent._ensure_required_tool_calls(calls, plan, "Compare Delhi today and Mumbai tomorrow")
    assert [call.args["date"] for call in result] == ["today", "tomorrow"]
