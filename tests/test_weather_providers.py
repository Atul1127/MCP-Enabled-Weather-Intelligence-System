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
    # Both forecast providers were tried (Open-Meteo, then the MET Norway fallback).
    assert body["provider"] == "api.open-meteo.com + api.met.no" and body["provider_status"] == 429


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


@pytest.fixture(autouse=True)
def clear_forecast_cache():
    weather_client._FORECAST_CACHE.clear()
    yield
    weather_client._FORECAST_CACHE.clear()


def test_forecast_is_cached_per_location(monkeypatch):
    get, calls = _fake_get(nominatim=(200, []))
    monkeypatch.setattr(weather_client.requests, "get", get)
    weather_client.fetch_weather(22.5726, 88.3639)
    weather_client.fetch_weather(22.5701, 88.3612)  # same ~1 km grid cell
    assert len(calls) == 1


def test_paid_open_meteo_key_uses_customer_api(monkeypatch):
    seen = {}

    def get(url, params=None, headers=None, timeout=None):
        seen.update(url=url, params=params)
        return FakeResponse(200, {"current": {}})

    monkeypatch.setenv("OPEN_METEO_API_KEY", "abc123")
    monkeypatch.setattr(weather_client.requests, "get", get)
    weather_client.fetch_weather(19.07, 72.87)
    assert seen["url"].startswith("https://customer-api.open-meteo.com/")
    assert seen["params"]["apikey"] == "abc123"


# --------------------------------------------------------------------------- MET Norway fallback

def _met_payload():
    """Shape of api.met.no locationforecast 2.0 /complete: hourly steps, then 6-hourly."""
    series = []
    start = weather_client.datetime(2026, 11, 25, 12, tzinfo=weather_client.timezone.utc)  # 17:30 IST
    for hour in range(0, 60):
        stamp = start + weather_client.timedelta(hours=hour)
        series.append({"time": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"), "data": {
            "instant": {"details": {"air_temperature": 30.0 + (hour % 5), "relative_humidity": 70.0,
                                    "wind_speed": 5.0, "wind_from_direction": 180.0, "cloud_area_fraction": 40.0}},
            "next_1_hours": {"summary": {"symbol_code": "rainshowers_day" if hour == 20 else "partlycloudy_day"},
                             "details": {"precipitation_amount": 2.0 if hour == 20 else 0.0, "probability_of_precipitation": 80.0 if hour == 20 else 5.0}},
            "next_6_hours": {"summary": {"symbol_code": "cloudy"}, "details": {"precipitation_amount": 9.9}},
        }})
    for block in range(0, 20):
        stamp = start + weather_client.timedelta(hours=60 + 6 * block)
        series.append({"time": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"), "data": {
            "instant": {"details": {"air_temperature": 28.0, "relative_humidity": 60.0, "wind_speed": 12.0}},
            "next_6_hours": {"summary": {"symbol_code": "heavyrainandthunder"},
                             "details": {"precipitation_amount": 8.0, "air_temperature_max": 33.0, "air_temperature_min": 24.0}},
        }})
    return {"type": "Feature", "properties": {"timeseries": series}}


def test_met_norway_is_converted_to_open_meteo_shape():
    data = weather_client._convert_met_norway(_met_payload())
    current, hourly, daily = data["current"], data["hourly"], data["daily"]
    assert current["time"] == "2026-11-25T17:30"            # UTC -> IST
    assert current["wind_speed_10m"] == 18.0                 # 5 m/s -> km/h
    assert current["weather_code"] == 2 and current["apparent_temperature"] > 30
    assert len(hourly["time"]) == 60 and hourly["precipitation_probability"][20] == 80.0
    assert len(daily["time"]) == 7 and daily["time"][0] == "2026-11-25"
    # Hourly blocks are summed once (the overlapping 6-hour totals are not double counted).
    assert daily["precipitation_sum"][1] == 2.0
    assert daily["precipitation_probability_max"][1] == 80.0
    assert daily["weather_code"][-1] == 95                  # thunder in the 6-hourly range
    assert daily["temperature_2m_max"][-1] == 33.0 and daily["temperature_2m_min"][-1] == 24.0
    assert data["attribution"].startswith("Weather data from MET Norway")


def test_open_meteo_429_falls_back_to_met_norway(monkeypatch):
    seen = []

    def get(url, params=None, headers=None, timeout=None):
        seen.append((url, params, headers))
        if "api.open-meteo.com" in url:
            return FakeResponse(429, {"reason": "Daily API request limit exceeded"})
        return FakeResponse(200, _met_payload())

    monkeypatch.setattr(weather_client.requests, "get", get)
    data = weather_client.fetch_weather(22.572645, 88.363892)
    assert data["source"] == "met.no" and data["current"]["temperature_2m"] == 30.0
    met_url, met_params, met_headers = seen[-1]
    assert "api.met.no" in met_url and met_params == {"lat": 22.5726, "lon": 88.3639}  # max 4 decimals
    assert "mcp-weather-intelligence" in met_headers["User-Agent"]                     # required by MET terms


def test_both_forecast_providers_down_names_both(monkeypatch):
    def get(url, params=None, headers=None, timeout=None):
        return FakeResponse(429 if "open-meteo" in url else 503, {})

    monkeypatch.setattr(weather_client.requests, "get", get)
    with pytest.raises(weather_client.UpstreamError) as info:
        weather_client.fetch_weather(19.07, 72.87)
    assert "api.open-meteo.com" in info.value.service and "api.met.no" in info.value.service


def test_heavy_rain_alert_when_probability_unknown():
    alerts = weather_client.detect_alerts({"time": ["2026-11-26"], "precipitation_sum": [25.0],
                                           "precipitation_probability_max": [None]})
    assert alerts[0]["hazard"] == "Heavy rain" and alerts[0]["probability"] is None


def test_dashboard_credits_both_data_sources():
    html = app.app.test_client().get("/").get_data(as_text=True)
    assert "Open-Meteo" in html and "MET Norway" in html
