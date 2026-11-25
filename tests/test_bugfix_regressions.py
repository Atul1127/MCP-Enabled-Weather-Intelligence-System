"""Regression tests for routing, Gemini turn pairing, retrieval scoping, alerts and readiness."""
from __future__ import annotations

import asyncio
import json
import runpy
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai import types

import app
import weather_client
import weather_agent_core.agent as agent_module
import weather_agent_core.executor as executor_module
import weather_agent_core.synthesizer as synthesizer_module
from rag.pipeline import RAGPipeline
from weather_agent_core.agent import WeatherAgent
from weather_agent_core.router import classify

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- routing

@pytest.mark.parametrize("query", [
    "What is the weather in Mumbai?",
    "What is the temperature in Delhi?",
    "What is the visibility in Delhi right now?",
    "what are the conditions at pune",
])
def test_located_weather_questions_use_live_tools(query):
    assert classify(query) == "live_weather"


@pytest.mark.parametrize("query", [
    "What is a cyclone?",
    "How does fog reduce visibility for drivers?",
    "Which WMO weather code represents a thunderstorm?",
])
def test_conceptual_questions_stay_knowledge(query):
    assert classify(query) == "knowledge"


# --------------------------------------------------------------------------- fallback args

@pytest.mark.parametrize("query, expected", [
    ("What is the current weather in Mumbai and what outdoor risks should I consider?", "Mumbai"),
    ("Will it rain in Pune in the evening tomorrow?", "Pune"),
    ("Is it safe for cycling in Kochi today?", "Kochi"),
    ("Forecast for Jaipur tomorrow", "Jaipur"),
    ("Best time to go out?\nLocation: Shimla", "Shimla"),
])
def test_fallback_location_extraction(query, expected):
    assert WeatherAgent._extract_location(query) == expected


def test_fallback_never_geocodes_the_whole_question():
    assert WeatherAgent._infer_tool_args("get_weather", "Should I go for a run?") is None


def test_date_hint_is_not_forced_when_both_days_are_mentioned():
    call = types.FunctionCall(name="get_forecast", args={"location": "Delhi", "date": "today"})
    WeatherAgent._apply_date_hint([call], "Compare today vs tomorrow in Delhi")
    assert call.args["date"] == "today"
    WeatherAgent._apply_date_hint([call], "Rain in Delhi tomorrow?")
    assert call.args["date"] == "tomorrow"


# --------------------------------------------------------------------------- Gemini turn pairing

def _assert_valid_gemini_history(contents):
    """Gemini requires each function-response turn to answer exactly the preceding model calls."""
    for index, content in enumerate(contents):
        responses = [p.function_response for p in (content.parts or []) if p.function_response]
        if not responses:
            continue
        assert index > 0, "function responses without a preceding model turn"
        previous = contents[index - 1]
        assert previous.role == "model", "function responses must follow a model turn"
        calls = [p.function_call for p in (previous.parts or []) if p.function_call]
        assert [c.name for c in calls] == [r.name for r in responses]


def _tool(name):
    return {"type": "function", "function": {"name": name, "description": name, "parameters": {
        "type": "object", "properties": {"location": {"type": "string"}, "query": {"type": "string"},
                                         "date": {"type": "string"}, "activity": {"type": "string"}}}}}


def _response(*calls):
    parts = [types.Part(function_call=call) for call in calls] or [types.Part(text="done")]
    return SimpleNamespace(function_calls=list(calls) or None,
                           candidates=[SimpleNamespace(content=types.Content(role="model", parts=parts))])


def _run_agent(monkeypatch, query, scripted_responses, tool_results):
    seen_contents = []
    scripted = list(scripted_responses)

    @asynccontextmanager
    async def fake_connect(trace_id=None):
        yield object()

    async def fake_discover(session):
        return [_tool(name) for name in sorted(agent_module.ALLOWED_TOOLS)]

    async def fake_call_tool(session, name, args):
        result = tool_results[name]
        return result.pop(0) if isinstance(result, list) else result

    def fake_generate_with_tools(contents, **kwargs):
        seen_contents.append(list(contents))
        _assert_valid_gemini_history(list(contents))
        return scripted.pop(0) if scripted else _response()

    def fake_structured(prompt, **kwargs):
        return json.dumps({"answer": "ok", "confidence": 0.8, "citations": [], "warnings": []})

    monkeypatch.setattr(agent_module, "connect", fake_connect)
    monkeypatch.setattr(agent_module, "discover_tools", fake_discover)
    monkeypatch.setattr(agent_module, "generate_with_tools", fake_generate_with_tools)
    monkeypatch.setattr(executor_module, "call_tool", fake_call_tool)
    monkeypatch.setattr(synthesizer_module, "generate_structured", fake_structured)
    result = asyncio.run(WeatherAgent(max_rounds=4, max_retries=1).run(query))
    return result, seen_contents


def test_orchestrator_added_calls_are_not_sent_to_gemini_as_responses(monkeypatch):
    # The model picks the wrong tool; the orchestrator adds the required one, which
    # fails once, forcing a second Gemini round with the earlier history.
    result, seen = _run_agent(
        monkeypatch,
        "Is it safe to play cricket in Chennai tomorrow?",
        [
            _response(types.FunctionCall(name="get_weather", args={"location": "Chennai"})),
            _response(types.FunctionCall(name="assess_weather_risk", args={"location": "Chennai", "activity": "cricket"})),
        ],
        {
            "get_weather": {"success": True, "current": {}},
            "assess_weather_risk": [
                {"success": False, "error": "boom", "error_type": "mcp_error"},
                {"success": True, "risk_level": "LOW"},
            ],
        },
    )
    assert len(seen) == 2
    assert result["success"] is True


def test_direct_rag_retry_does_not_send_orphan_function_responses(monkeypatch):
    # Empty retrieval makes the verifier ask Gemini for recovery after the
    # planner-executed (not model-requested) search_weather call.
    result, seen = _run_agent(
        monkeypatch,
        "What does precipitation probability actually mean?",
        [_response()],
        {"search_weather": {"success": True, "documents": [], "sources": [], "context": ""}},
    )
    assert len(seen) == 1
    assert all(not any(p.function_response for p in (c.parts or [])) for c in seen[0])


# --------------------------------------------------------------------------- retrieval

def test_unfiltered_retrieval_does_not_materialize_every_row_id():
    class Store:
        def filtered_rows(self, **kwargs):
            raise AssertionError("filtered_rows must not run without filters")

        def bm25_search(self, query, limit, allowed=None):
            assert allowed is None
            return [{"id": "d1", "narrative_text": "Heavy rain forms when...", "source": "guide"}]

    pipeline = RAGPipeline()
    pipeline._store = Store()
    result = pipeline.retrieve("What causes heavy rain?", top_k=3)
    assert [doc["id"] for doc in result.documents] == ["d1"]


def test_postgres_scope_excludes_synced_live_documents(monkeypatch):
    from rag.postgres_store import PostgresRagStore
    monkeypatch.delenv("WEATHER_RAG_INCLUDE_LIVE_DOCS", raising=False)
    clause, params = PostgresRagStore()._scope_clause(None)
    assert "NOT IN" in clause and params == ["current", "forecast"]


# --------------------------------------------------------------------------- alerts

def test_alert_detection_tolerates_missing_values():
    daily = {
        "time": ["2026-11-25", "2026-11-26"],
        "precipitation_probability_max": [None, 80],
        "precipitation_sum": [None, 20.0],
        "wind_speed_10m_max": [None, 10],
        "apparent_temperature_max": [None, 41],
        "weather_code": [None, 95],
    }
    alerts = weather_client.detect_alerts(daily)
    assert {a["hazard"] for a in alerts} == {"Heavy rain", "Extreme heat", "Thunderstorm"}
    assert all(a["date"] == "2026-11-26" for a in alerts)
    assert weather_client.highest_severity(alerts) == "HIGH"
    assert weather_client.highest_severity([]) == "NONE"


# --------------------------------------------------------------------------- readiness / deploy

def test_readyz_reports_unavailable_store(monkeypatch):
    class Broken:
        def check_ready(self):
            raise RuntimeError("database down")

    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(app.rag_service, "get_rag_pipeline", lambda: Broken())
    response = app.app.test_client().get("/readyz")
    assert response.status_code == 503
    assert response.get_json()["checks"]["rag_store"] is False


def test_gunicorn_binds_platform_port(monkeypatch):
    monkeypatch.setenv("PORT", "10000")
    assert runpy.run_path(str(ROOT / "docker" / "gunicorn.conf.py"))["bind"].endswith(":10000")
    monkeypatch.delenv("PORT")
    monkeypatch.delenv("FLASK_RUN_PORT", raising=False)
    assert runpy.run_path(str(ROOT / "docker" / "gunicorn.conf.py"))["bind"].endswith(":8000")


def test_new_gemini_models_keep_default_temperature(monkeypatch):
    import services.llm as llm
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "low")
    assert llm._config_for_model("gemini-3.8-flash", temperature=0.0).temperature is None
    assert llm._config_for_model("gemini-4-argon", temperature=0.0).temperature is None
    assert llm._config_for_model("gemini-2.5-flash", temperature=0.0).temperature == 0.0
