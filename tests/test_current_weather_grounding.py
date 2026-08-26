import mcp_server


def test_current_time_of_day_uses_live_sunrise_sunset():
    daily = {
        "time": ["2026-08-26"],
        "sunrise": ["2026-08-26T05:30"],
        "sunset": ["2026-08-26T18:20"],
    }

    assert mcp_server._current_time_of_day("2026-08-26T12:00", daily) == "day"
    assert mcp_server._current_time_of_day("2026-08-26T20:00", daily) == "night"


def test_current_summary_preserves_observation_condition():
    current = {
        "time": "2026-08-26T21:00",
        "weather_code": 2,
        "temperature_2m": 28.6,
        "apparent_temperature": 31.0,
        "relative_humidity_2m": 89,
        "cloud_cover": 75,
        "precipitation": 0.0,
        "wind_speed_10m": 4.0,
        "wind_direction_10m": 45,
    }
    daily = {
        "time": ["2026-08-26"],
        "sunrise": ["2026-08-26T05:30"],
        "sunset": ["2026-08-26T18:20"],
    }

    summary = mcp_server._current_summary(current, daily, "Asia/Kolkata")

    assert summary["time_of_day"] == "night"
    assert summary["condition"] == "Partly cloudy"
    assert summary["temperature_c"] == 28.6
    assert summary["relative_humidity_pct"] == 89
    assert summary["timezone"] == "Asia/Kolkata"
