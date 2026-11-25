"""Per-client rate limiting on the Gemini-backed endpoints."""
import pytest

import app as flask_app
from services.rate_limit import parse_limit


def test_parse_limit():
    assert parse_limit("10/minute") == (10, 60)
    assert parse_limit("5/hours") == (5, 3600)
    assert parse_limit("0") is None
    with pytest.raises(ValueError):
        parse_limit("10/fortnight")


def test_agent_endpoint_returns_429_after_limit(monkeypatch):
    monkeypatch.setenv("WEATHER_AGENT_RATE_LIMIT", "2/minute")
    client = flask_app.app.test_client()
    # An empty query is rejected with 400 before the agent runs, but still counts.
    codes = [client.post("/weather/agent", json={"query": ""}).status_code for _ in range(3)]
    assert codes == [400, 400, 429]
    response = client.post("/weather/agent", json={"query": ""})
    assert response.status_code == 429 and int(response.headers["Retry-After"]) >= 1


def test_limits_are_per_client(monkeypatch):
    monkeypatch.setenv("WEATHER_AGENT_RATE_LIMIT", "1/minute")
    monkeypatch.setenv("WEATHER_TRUST_PROXY", "1")
    client = flask_app.app.test_client()
    first = client.post("/weather/agent", json={"query": ""}, headers={"X-Forwarded-For": "1.1.1.1"})
    other = client.post("/weather/agent", json={"query": ""}, headers={"X-Forwarded-For": "2.2.2.2"})
    again = client.post("/weather/agent", json={"query": ""}, headers={"X-Forwarded-For": "1.1.1.1"})
    assert (first.status_code, other.status_code, again.status_code) == (400, 400, 429)


def test_limit_can_be_disabled(monkeypatch):
    monkeypatch.setenv("WEATHER_AGENT_RATE_LIMIT", "0")
    client = flask_app.app.test_client()
    assert all(client.post("/weather/agent", json={"query": ""}).status_code == 400 for _ in range(5))
