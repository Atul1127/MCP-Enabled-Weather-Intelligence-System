"""Indian Weather Intelligence MCP server.

The server exposes deterministic live-weather tools plus the advanced local RAG
engine. Agent orchestration remains outside the server boundary.
"""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer

import advanced_rag
import lakebase
import weather_client

mcp = MCPServer("indian-weather-intelligence")


@mcp.tool()
def get_weather(location: str) -> dict[str, Any]:
    """Get current weather and a 7-day forecast for an Indian location."""
    location = location.strip() if location else ""
    if not location:
        raise ValueError("location cannot be empty")
    details = weather_client.geocode_location_details(location)
    if not details:
        return {"success": False, "error": f"Could not resolve location: {location}"}
    weather = weather_client.fetch_weather(details["latitude"], details["longitude"])
    return {"success": True, "location": details, "current": weather.get("current", {}), "daily": weather.get("daily", {})}


@mcp.tool()
def get_weather_alerts(location: str) -> dict[str, Any]:
    """Detect actionable hazards from the live 7-day forecast."""
    weather = get_weather(location)
    if not weather.get("success"):
        return weather
    daily = weather.get("daily", {})
    alerts: list[dict[str, Any]] = []
    dates = daily.get("time", [])
    for i, date in enumerate(dates):
        probability = (daily.get("precipitation_probability_max") or [0])[i] if i < len(daily.get("precipitation_probability_max") or []) else 0
        rain = (daily.get("precipitation_sum") or [0])[i] if i < len(daily.get("precipitation_sum") or []) else 0
        wind = (daily.get("wind_speed_10m_max") or [0])[i] if i < len(daily.get("wind_speed_10m_max") or []) else 0
        apparent = (daily.get("apparent_temperature_max") or [0])[i] if i < len(daily.get("apparent_temperature_max") or []) else 0
        code = (daily.get("weather_code") or [0])[i] if i < len(daily.get("weather_code") or []) else 0
        if probability >= 70 and rain >= 15:
            alerts.append({"date": date, "severity": "HIGH", "hazard": "Heavy rain", "probability": probability, "details": f"{rain:.1f} mm expected with {probability}% precipitation probability."})
        elif probability >= 70:
            alerts.append({"date": date, "severity": "MODERATE", "hazard": "High rain probability", "probability": probability, "details": f"Precipitation probability is {probability}%."})
        if wind >= 40:
            alerts.append({"date": date, "severity": "HIGH", "hazard": "Strong wind", "details": f"Maximum forecast wind is {wind:.1f} km/h."})
        if apparent >= 40:
            alerts.append({"date": date, "severity": "HIGH", "hazard": "Extreme heat", "details": f"Maximum apparent temperature is {apparent:.1f}°C."})
        elif apparent >= 35:
            alerts.append({"date": date, "severity": "MODERATE", "hazard": "High heat", "details": f"Maximum apparent temperature is {apparent:.1f}°C."})
        if code >= 95:
            alerts.append({"date": date, "severity": "HIGH", "hazard": "Thunderstorm", "details": f"Forecast weather code is {code}."})
    alerts.sort(key=lambda x: (0 if x["severity"] == "HIGH" else 1, x["date"]))
    return {"success": True, "location": weather["location"], "alerts": alerts, "alert_count": len(alerts), "highest_severity": alerts[0]["severity"] if alerts else "NONE", "disclaimer": "Application-level forecast hazard detection; not an official government warning."}


@mcp.tool()
def assess_weather_risk(location: str, activity: str = "outdoor activity") -> dict[str, Any]:
    """Assess practical activity risk using live forecast signals."""
    weather = get_weather(location)
    if not weather.get("success"):
        return weather
    current = weather.get("current", {})
    daily = weather.get("daily", {})
    rain_probability = (daily.get("precipitation_probability_max") or [0])[0]
    precipitation = (daily.get("precipitation_sum") or [0])[0]
    wind = (daily.get("wind_speed_10m_max") or [0])[0]
    apparent = (daily.get("apparent_temperature_max") or [0])[0]
    code = (daily.get("weather_code") or [0])[0]
    score = 0
    factors: list[str] = []
    if rain_probability >= 70: score += 2; factors.append(f"high precipitation probability ({rain_probability}%)")
    elif rain_probability >= 40: score += 1; factors.append(f"moderate precipitation probability ({rain_probability}%)")
    if precipitation >= 10: score += 2; factors.append(f"heavy expected precipitation ({precipitation:.1f} mm)")
    elif precipitation >= 3: score += 1; factors.append(f"expected precipitation ({precipitation:.1f} mm)")
    if wind >= 30: score += 2; factors.append(f"strong wind ({wind:.1f} km/h)")
    elif wind >= 20: score += 1; factors.append(f"elevated wind ({wind:.1f} km/h)")
    if apparent >= 40: score += 2; factors.append(f"very high apparent temperature ({apparent:.1f}°C)")
    elif apparent >= 35: score += 1; factors.append(f"high apparent temperature ({apparent:.1f}°C)")
    if code >= 95: score += 2; factors.append("thunderstorm risk in the forecast")
    risk = "HIGH" if score >= 5 else "MODERATE" if score >= 2 else "LOW"
    recommendation = f"Avoid or postpone the {activity} if possible." if risk == "HIGH" else f"The {activity} is possible with precautions and a backup plan." if risk == "MODERATE" else f"Conditions look generally suitable for the {activity}."
    return {"success": True, "location": weather["location"], "activity": activity, "risk_level": risk, "score": score, "recommendation": recommendation, "factors": factors or ["no major weather risk detected"], "data": {"rain_probability": rain_probability, "precipitation_mm": precipitation, "max_wind_kmh": wind, "apparent_temperature_max_c": apparent, "weather_code": code, "current_temperature_c": current.get("temperature_2m")}}


@mcp.tool()
def search_weather(query: str, top_k: int = 5, location: str | None = None, state: str | None = None) -> dict[str, Any]:
    """Search weather knowledge with multi-query dense+BM25 RRF and reranking."""
    return advanced_rag.answer(query.strip(), top_k=max(1, min(20, int(top_k))), location=location, state=state) if query and query.strip() else {"success": False, "error": "query cannot be empty"}


@mcp.tool()
def ask_weather(query: str, top_k: int = 5, location: str | None = None, state: str | None = None) -> dict[str, Any]:
    """Generate a grounded weather answer with advanced local RAG."""
    if not query or not query.strip():
        raise ValueError("query cannot be empty")
    return advanced_rag.answer(query.strip(), top_k=max(1, min(20, int(top_k))), location=location, state=state)


@mcp.tool()
def sync_weather(locations: list[str]) -> dict[str, Any]:
    """Fetch and store fresh weather data for Indian locations."""
    cleaned = [x.strip() for x in locations if x and x.strip()] if locations else []
    if not cleaned:
        raise ValueError("locations cannot be empty")
    count = weather_client.sync_locations(cleaned)
    return {"success": True, "locations": cleaned, "documents_synced": count}


@mcp.tool()
def database_health() -> dict[str, Any]:
    """Check PostgreSQL/Lakebase reachability."""
    try:
        connected = lakebase.check_connection()
        return {"success": connected, "backend": lakebase.DATABASE_BACKEND, "status": "ok" if connected else "unavailable"}
    except Exception as exc:
        return {"success": False, "backend": lakebase.DATABASE_BACKEND, "status": "error", "error": str(exc)}


if __name__ == "__main__":
    mcp.run()
