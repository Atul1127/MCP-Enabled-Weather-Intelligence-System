from weather_agent_core.router import classify


def test_precipitation_probability_meaning_is_knowledge():
    assert classify("What does precipitation probability actually mean?") == "knowledge"


def test_live_forecast_stays_live():
    assert classify("What is the forecast for Delhi tomorrow?") == "live_weather"


def test_risk_query_stays_risk():
    assert classify("Should I run outdoors in Mumbai tomorrow?") == "activity_risk"


def test_alert_risk_word_is_not_activity_risk():
    assert classify("What weather risks should I watch for in Kolkata?") == "alerts"


def test_conceptual_outdoor_question_is_knowledge():
    assert classify("How can heavy rain affect outdoor activities?") == "knowledge"


def test_precaution_questions_are_knowledge():
    assert classify("What precautions are useful during extreme heat?") == "knowledge"
    assert classify("What precautions should people take during thunderstorms?") == "knowledge"


def test_risk_markers_do_not_match_inside_words():
    assert classify("Display the forecast for Delhi tomorrow") == "live_weather"
    assert classify("Runway weather conditions for Mumbai tomorrow") == "live_weather"
