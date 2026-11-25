"""Provider failures: geocoding fallback, clear upstream errors, no endless agent retries."""
from __future__ import annotations

import asyncio

import pytest
import requests

import app
import weather_client
from weather_agent_core.graph import build_weather_graph


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def clear_geocode_cache():
    weather_client._GEOCODE_CACHE.clear()
    yield
    weather_client._GEOCODE_CACHE.clear()


OPEN_METEO_GEO = {"results": [{"name": "Kolkata", "latitude": 22.57, "longitude": 88.36, "admin1": "West Bengal", "admin2": "Kolkata", "country": "India"}]}


def _fake_get(nominatim, open_meteo_geo=(200, OPEN_METEO_GEO), forecast=(200, {"current": {"temperature_2m": 30}, "daily": {}, "hourly": {}})):
    calls = []

    def get(url, params=None, headers=None, timeout=None):
        calls.append((url, headers))
        if "nominatim" in url:
            if isinstance(nominatim, Exception):
                raise nominatim
            return FakeResponse(*nominatim)
        if "geocoding-api.open-meteo" in url:
            return FakeResponse(*open_meteo_geo)
        return FakeResponse(*forecast)

    return get, calls


def test_nominatim_block_falls_back_to_open_meteo(monkeypatch):
    get, calls = _fake_get(nominatim=(403, {"error": "blocked"}))
    monkeypatch.setattr(weather_client.requests, "get", get)
    details = weather_client.geocode_location_details("Kolkata")
    assert details["latitude"] == 22.57 and details["state"] == "West Bengal"
    assert "mcp-weather-intelligence" in calls[0][1]["User-Agent"]


def test_nominatim_empty_answer_falls_back(monkeypatch):
    get, _ = _fake_get(nominatim=(200, []))
    monkeypatch.setattr(weather_client.requests, "get", get)
    assert weather_client.geocode_location_details("Kolkata, West Bengal")["longitude"] == 88.36


def test_geocoding_is_cached(monkeypatch):
    get, calls = _fake_get(nominatim=(200, [{"lat": "22.5", "lon": "88.3", "display_name": "Kolkata", "address": {}}]))
    monkeypatch.setattr(weather_client.requests, "get", get)
    weather_client.geocode_location_details("Kolkata")
    weather_client.geocode_location_details("kolkata")
    assert len(calls) == 1


def test_all_geocoders_down_raises_upstream_error(monkeypatch):
    get, _ = _fake_get(nominatim=requests.ConnectionError("down"), open_meteo_geo=(429, {}))
    monkeypatch.setattr(weather_client.requests, "get", get)
    with pytest.raises(weather_client.UpstreamError) as info:
        weather_client.geocode_location_details("Kolkata")
    assert info.value.status == 429


def test_unknown_place_returns_none(monkeypatch):
    get, _ = _fake_get(nominatim=(200, []), open_meteo_geo=(200, {}))
    monkeypatch.setattr(weather_client.requests, "get", get)
    assert weather_client.geocode_location_details("Nowhereville") is None


def test_current_weather_reports_failing_provider(monkeypatch):
    get, _ = _fake_get(nominatim=(200, [{"lat": "22.5", "lon": "88.3", "address": {}}]), forecast=(429, {}))
    monkeypatch.setattr(weather_client.requests, "get", get)
    response = app.app.test_client().post("/weather/current", json={"location": "Kolkata"})
    assert response.status_code == 502
    body = response.get_json()
    assert body["provider"] == "api.open-meteo.com" and body["provider_status"] == 429


def test_current_weather_succeeds_through_fallback(monkeypatch):
    get, _ = _fake_get(nominatim=(403, {}))
    monkeypatch.setattr(weather_client.requests, "get", get)
    response = app.app.test_client().post("/weather/current", json={"location": "Kolkata"})
    assert response.status_code == 200
    assert response.get_json()["current"]["temperature_2m"] == 30


def test_provider_outage_stops_agent_retries():
    async def run():
        seen = []
        failed = [{"tool": "get_weather", "result": {"success": False, "error_type": "upstream_unavailable"}}]
        plan = {"steps": [{"preferred_tools": ["get_weather"], "required": True}]}
        async def router(state): return {}
        async def planner(state): return {"plan": plan}
        async def reasoner(state):
            seen.append("reasoner")
            return {"next_action": "tool", "rounds": int(state.get("rounds", 0)) + 1, "pending_calls": ["x"]}
        async def executor(state): return {"observations": failed}
        async def verifier(state): return {"verification": {"sufficient": False}, "retry_count": 1}
        async def synthesizer(state): return {"answer": "provider down"}
        graph = build_weather_graph(router=router, planner=planner, reasoner=reasoner, executor=executor,
                                    verifier=verifier, synthesizer=synthesizer, max_rounds=4, max_retries=1)
        result = await graph.ainvoke({"rounds": 0})
        assert result["answer"] == "provider down"
        assert seen == ["reasoner"]  # one Gemini round, no re-planning loop
    asyncio.run(run())
