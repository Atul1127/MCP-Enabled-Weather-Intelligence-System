import asyncio

import app
import llm_provider


def test_agent_endpoint_rejects_empty_query():
    response = app.app.test_client().post("/weather/agent", json={"query": ""})
    assert response.status_code == 400
    assert "query" in response.get_json()["error"]


def test_agent_endpoint_uses_canonical_agent(monkeypatch):
    expected = {"success": True, "answer": "ok", "trace_id": "test"}

    async def fake_run(self, query):
        assert query == "Weather in Kolkata"
        return expected

    monkeypatch.setattr(app.WeatherAgent, "run", fake_run)
    response = app.app.test_client().post(
        "/weather/agent",
        json={"query": "Weather in Kolkata"},
    )
    assert response.status_code == 200
    assert response.get_json() == expected


def test_agent_endpoint_returns_gateway_error_without_leaking_details(monkeypatch):
    async def fake_run(self, query):
        raise RuntimeError("secret database URL")

    monkeypatch.setattr(app.WeatherAgent, "run", fake_run)
    response = app.app.test_client().post(
        "/weather/agent",
        json={"query": "Weather in Kolkata"},
    )
    assert response.status_code == 502
    assert response.get_json()["error"] == "Failed to generate agent answer"
    assert "secret database URL" not in response.get_data(as_text=True)


def test_gemini_model_fallback_order(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "fallback-a,fallback-b,primary")
    assert llm_provider._gemini_models() == ["primary", "fallback-a", "fallback-b"]


def test_gemini_retryable_errors():
    assert llm_provider._gemini_retryable(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert llm_provider._gemini_retryable(RuntimeError("503 UNAVAILABLE"))
    assert not llm_provider._gemini_retryable(RuntimeError("invalid prompt"))
