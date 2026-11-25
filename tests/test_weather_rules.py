"""Shared hazard rules: IMD-aligned labels, one wind threshold, same output everywhere."""
import weather_rules


def _daily(**columns):
    days = len(columns.get("time", ["2026-07-01"]))
    base = {
        "time": ["2026-07-01"],
        "precipitation_probability_max": [0] * days,
        "precipitation_sum": [0] * days,
        "wind_speed_10m_max": [0] * days,
        "temperature_2m_max": [30] * days,
        "apparent_temperature_max": [30] * days,
        "weather_code": [0] * days,
    }
    base.update(columns)
    return base


def _hazards(daily):
    return [alert["hazard"] for alert in weather_rules.detect_alerts(daily)]


def test_15mm_is_not_called_heavy_rain():
    alerts = weather_rules.detect_alerts(_daily(precipitation_probability_max=[80], precipitation_sum=[20]))
    assert [(a["severity"], a["hazard"]) for a in alerts] == [("MODERATE", "Significant rain likely")]


def test_imd_rain_categories():
    assert _hazards(_daily(precipitation_sum=[70])) == ["Heavy rain (IMD category)"]
    assert _hazards(_daily(precipitation_sum=[120])) == ["Very heavy rain (IMD category)"]
    assert _hazards(_daily(precipitation_sum=[210])) == ["Extremely heavy rain (IMD category)"]


def test_heatwave_uses_actual_temperature_and_feels_like_is_heat_stress():
    assert _hazards(_daily(temperature_2m_max=[45.5], apparent_temperature_max=[44])) == ["Heatwave conditions (IMD absolute threshold)"]
    assert _hazards(_daily(temperature_2m_max=[38], apparent_temperature_max=[41])) == ["High heat stress (feels like)"]
    assert _hazards(_daily(temperature_2m_max=[34], apparent_temperature_max=[36])) == ["Heat stress (feels like)"]


def test_alerts_sorted_high_first_then_date():
    daily = _daily(
        time=["2026-07-01", "2026-07-02"],
        precipitation_probability_max=[80, 0],
        precipitation_sum=[0, 0],
        wind_speed_10m_max=[0, 45],
        temperature_2m_max=[30, 30],
        apparent_temperature_max=[30, 30],
        weather_code=[0, 0],
    )
    alerts = weather_rules.detect_alerts(daily)
    assert [(a["date"], a["severity"]) for a in alerts] == [("2026-07-02", "HIGH"), ("2026-07-01", "MODERATE")]
    assert all(a["recommendation"] for a in alerts)


def test_wind_threshold_matches_between_alerts_and_risk():
    just_below = {"max_wind_kmh": weather_rules.STRONG_WIND_KMH - 1}
    at_threshold = {"max_wind_kmh": weather_rules.STRONG_WIND_KMH}
    assert "strong wind" not in " ".join(weather_rules.assess_activity_risk(just_below, "cycling")["factors"])
    assert "strong wind" in " ".join(weather_rules.assess_activity_risk(at_threshold, "cycling")["factors"])
    assert _hazards(_daily(wind_speed_10m_max=[weather_rules.STRONG_WIND_KMH - 1])) == []
    assert _hazards(_daily(wind_speed_10m_max=[weather_rules.STRONG_WIND_KMH])) == ["Strong wind"]


def test_dashboard_and_agent_return_identical_alerts(monkeypatch):
    import app as flask_app
    import weather_client
    from mcp_integration import mcp_server

    daily = _daily(precipitation_probability_max=[90], precipitation_sum=[80], weather_code=[95])
    weather = {"current": {}, "daily": daily, "hourly": {}}
    details = {"latitude": 19.07, "longitude": 72.87, "display_name": "Mumbai", "state": "Maharashtra", "district": None}
    monkeypatch.setattr(weather_client, "geocode_location_details", lambda location: details)
    monkeypatch.setattr(weather_client, "fetch_weather", lambda lat, lon: weather)

    dashboard = flask_app.app.test_client().post("/weather/alerts", json={"location": "Mumbai"}).get_json()
    agent = mcp_server.get_weather_alerts("Mumbai")
    assert dashboard["alerts"] == agent["alerts"]
    assert dashboard["highest_severity"] == agent["highest_severity"] == "HIGH"
