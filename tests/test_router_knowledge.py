from weather_agent_core.router import classify


def test_precipitation_probability_meaning_is_knowledge():
    assert classify("What does precipitation probability actually mean?") == "knowledge"


def test_live_forecast_stays_live():
    assert classify("What is the forecast for Delhi tomorrow?") == "live_weather"


def test_risk_query_stays_risk():
    assert classify("Should I run outdoors in Mumbai tomorrow?") == "activity_risk"
