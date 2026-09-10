from weather_agent_core.router import classify


def test_precipitation_probability_meaning_is_knowledge():
    assert classify("What does precipitation probability actually mean?") == "knowledge"


def test_live_forecast_stays_live():
    assert classify("What is the forecast for Delhi tomorrow?") == "live_weather"


def test_risk_query_stays_risk():
    assert classify("Should I run outdoors in Mumbai tomorrow?") == "activity_risk"


def test_risk_markers_do_not_match_inside_words():
    assert classify("Display the forecast for Delhi tomorrow") == "live_weather"
    assert classify("Runway weather conditions for Mumbai tomorrow") == "live_weather"


def test_heat_precautions_are_knowledge():
    assert classify("What precautions are useful during extreme heat?") == "knowledge"


def test_weather_hazards_are_alerts():
    assert classify("What weather hazards are expected tomorrow?") == "alerts"


def test_typical_risks_are_knowledge():
    assert classify("What are the typical risks of heavy rainfall?") == "knowledge"


def test_conceptual_safety_question_beats_activity_risk():
    assert classify("What safety measures help during a heatwave?") == "knowledge"
