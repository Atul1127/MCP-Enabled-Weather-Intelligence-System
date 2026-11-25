"""Shared forecast hazard and activity-risk rules.

Both the dashboard API (app.py) and the MCP server use these functions, so the
dashboard and the agent can never disagree about the same forecast.

Rain labels follow the India Meteorological Department (IMD) 24-hour rainfall
categories. IMD declares heatwaves from actual maximum temperature (and its
departure from the local normal, which this app does not have), so only the
IMD absolute thresholds are used for "heatwave". Apparent-temperature alerts
are labelled "heat stress" to keep them distinct from official IMD terms.
These are application-level forecast checks, not official warnings.
"""
from __future__ import annotations

from typing import Any

DISCLAIMER = "Application-level forecast hazard detection; not an official IMD or government warning."

# IMD 24-hour rainfall categories (mm).
IMD_HEAVY_RAIN_MM = 64.5
IMD_VERY_HEAVY_RAIN_MM = 115.6
IMD_EXTREMELY_HEAVY_RAIN_MM = 204.5
# Below IMD "heavy", but worth flagging when rain is very likely.
SIGNIFICANT_RAIN_MM = 15.0
RAIN_LIKELY_PROBABILITY_PCT = 70

# IMD absolute heatwave thresholds for the plains (actual maximum temperature).
IMD_HEATWAVE_MAX_C = 45.0
IMD_SEVERE_HEATWAVE_MAX_C = 47.0
# Feels-like temperature thresholds.
HIGH_HEAT_STRESS_APPARENT_C = 40.0
HEAT_STRESS_APPARENT_C = 35.0

# One wind threshold for both alerts and activity risk.
STRONG_WIND_KMH = 40.0
ELEVATED_WIND_KMH = 25.0

THUNDERSTORM_MIN_CODE = 95

_SEVERITY_ORDER = {"HIGH": 0, "MODERATE": 1, "LOW": 2}


def _value(series: Any, index: int) -> float:
    if not isinstance(series, list) or index >= len(series) or series[index] is None:
        return 0.0
    try:
        return float(series[index])
    except (TypeError, ValueError):
        return 0.0


def detect_alerts(daily: dict[str, Any]) -> list[dict[str, Any]]:
    """Return forecast hazards for every day in an Open-Meteo ``daily`` block,
    sorted by severity and then date."""
    alerts: list[dict[str, Any]] = []
    for index, date in enumerate(daily.get("time") or []):
        probability = _value(daily.get("precipitation_probability_max"), index)
        rain = _value(daily.get("precipitation_sum"), index)
        wind = _value(daily.get("wind_speed_10m_max"), index)
        max_temp = _value(daily.get("temperature_2m_max"), index)
        apparent = _value(daily.get("apparent_temperature_max"), index)
        code = int(_value(daily.get("weather_code"), index))

        def add(severity: str, hazard: str, details: str, recommendation: str, **extra: Any) -> None:
            alerts.append({"date": date, "severity": severity, "hazard": hazard, "details": details, "recommendation": recommendation, **extra})

        if rain >= IMD_EXTREMELY_HEAVY_RAIN_MM:
            add("HIGH", "Extremely heavy rain (IMD category)", f"{rain:.1f} mm forecast in 24 hours.", "Avoid travel where possible; flooding is likely.", probability=probability)
        elif rain >= IMD_VERY_HEAVY_RAIN_MM:
            add("HIGH", "Very heavy rain (IMD category)", f"{rain:.1f} mm forecast in 24 hours.", "Avoid unnecessary travel and low-lying areas.", probability=probability)
        elif rain >= IMD_HEAVY_RAIN_MM:
            add("HIGH", "Heavy rain (IMD category)", f"{rain:.1f} mm forecast in 24 hours.", "Plan for waterlogging and travel disruption.", probability=probability)
        elif probability >= RAIN_LIKELY_PROBABILITY_PCT and rain >= SIGNIFICANT_RAIN_MM:
            add("MODERATE", "Significant rain likely", f"{rain:.1f} mm expected with {probability:.0f}% probability.", "Keep rain protection and allow extra travel time.", probability=probability)
        elif probability >= RAIN_LIKELY_PROBABILITY_PCT:
            add("MODERATE", "Rain likely", f"Precipitation probability is {probability:.0f}%.", "Keep rain protection available and monitor the forecast.", probability=probability)

        if wind >= STRONG_WIND_KMH:
            add("HIGH", "Strong wind", f"Maximum forecast wind is {wind:.1f} km/h.", "Avoid exposed outdoor activities and secure loose objects.")

        if max_temp >= IMD_SEVERE_HEATWAVE_MAX_C:
            add("HIGH", "Severe heatwave conditions (IMD absolute threshold)", f"Maximum temperature is {max_temp:.1f}°C.", "Stay indoors during the afternoon, hydrate, and check on vulnerable people.")
        elif max_temp >= IMD_HEATWAVE_MAX_C:
            add("HIGH", "Heatwave conditions (IMD absolute threshold)", f"Maximum temperature is {max_temp:.1f}°C.", "Avoid outdoor exertion between 12 and 4 pm and hydrate.")
        elif apparent >= HIGH_HEAT_STRESS_APPARENT_C:
            add("HIGH", "High heat stress (feels like)", f"Maximum feels-like temperature is {apparent:.1f}°C.", "Limit strenuous outdoor activity, hydrate, and seek shade.")
        elif apparent >= HEAT_STRESS_APPARENT_C:
            add("MODERATE", "Heat stress (feels like)", f"Maximum feels-like temperature is {apparent:.1f}°C.", "Take heat precautions during strenuous outdoor activity.")

        if code >= THUNDERSTORM_MIN_CODE:
            add("HIGH", "Thunderstorm", f"Thunderstorm weather code {code} is forecast.", "Avoid exposed outdoor locations and seek sturdy shelter.")

    alerts.sort(key=lambda alert: (_SEVERITY_ORDER.get(alert["severity"], 9), str(alert["date"])))
    return alerts


def highest_severity(alerts: list[dict[str, Any]]) -> str:
    return alerts[0]["severity"] if alerts else "NONE"


def assess_activity_risk(forecast: dict[str, Any], activity: str) -> dict[str, Any]:
    """Score one day's forecast for an outdoor activity.

    ``forecast`` is the daily summary produced by the MCP server
    (precipitation_probability_pct, precipitation_mm, max_wind_kmh,
    apparent_temperature_max_c, weather_code).
    """
    rain_probability = float(forecast.get("precipitation_probability_pct") or 0)
    precipitation = float(forecast.get("precipitation_mm") or 0)
    wind = float(forecast.get("max_wind_kmh") or 0)
    apparent = float(forecast.get("apparent_temperature_max_c") or 0)
    code = int(forecast.get("weather_code") or 0)
    score = 0
    factors: list[str] = []

    if rain_probability >= RAIN_LIKELY_PROBABILITY_PCT:
        score += 2; factors.append(f"high precipitation probability ({rain_probability:.0f}%)")
    elif rain_probability >= 40:
        score += 1; factors.append(f"moderate precipitation probability ({rain_probability:.0f}%)")
    if precipitation >= SIGNIFICANT_RAIN_MM:
        score += 2; factors.append(f"significant expected rain ({precipitation:.1f} mm)")
    elif precipitation >= 3:
        score += 1; factors.append(f"some expected rain ({precipitation:.1f} mm)")
    if wind >= STRONG_WIND_KMH:
        score += 2; factors.append(f"strong wind ({wind:.1f} km/h)")
    elif wind >= ELEVATED_WIND_KMH:
        score += 1; factors.append(f"elevated wind ({wind:.1f} km/h)")
    if apparent >= HIGH_HEAT_STRESS_APPARENT_C:
        score += 2; factors.append(f"high heat stress (feels like {apparent:.1f}°C)")
    elif apparent >= HEAT_STRESS_APPARENT_C:
        score += 1; factors.append(f"heat stress (feels like {apparent:.1f}°C)")
    if code >= THUNDERSTORM_MIN_CODE:
        score += 2; factors.append("thunderstorm risk in the forecast")

    risk = "HIGH" if score >= 5 else "MODERATE" if score >= 2 else "LOW"
    if risk == "HIGH":
        recommendation = f"Avoid or postpone the {activity} if possible."
    elif risk == "MODERATE":
        recommendation = f"The {activity} is possible with precautions and a backup plan."
    else:
        recommendation = f"Conditions look generally suitable for the {activity}."
    return {"risk_level": risk, "score": score, "recommendation": recommendation, "factors": factors or ["no major weather risk detected"]}
