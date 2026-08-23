"""
Indian Weather Intelligence MCP Server.

Exposes the project's weather and RAG capabilities through the
Model Context Protocol. The server is intentionally domain-focused:
weather is the tool ecosystem, while MCP is the integration layer.

Tools:
    get_weather
    assess_weather_risk
    search_weather
    ask_weather
    sync_weather
    database_health

Transport:
    stdio

No paid API key is required.
"""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer

import lakebase
import rag_service
import weather_client


mcp = MCPServer("indian-weather-intelligence")


@mcp.tool()
def get_weather(location: str) -> dict[str, Any]:
    """Get current weather and a 7-day forecast for an Indian location."""
    if not location or not location.strip():
        raise ValueError("location cannot be empty")

    location = location.strip()
    details = weather_client.geocode_location_details(location)
    if not details:
        return {"success": False, "error": f"Could not resolve location: {location}"}

    weather = weather_client.fetch_weather(details["latitude"], details["longitude"])
    return {
        "success": True,
        "location": details,
        "current": weather.get("current", {}),
        "daily": weather.get("daily", {}),
    }


@mcp.tool()
def assess_weather_risk(
    location: str,
    activity: str = "outdoor activity",
) -> dict[str, Any]:
    """Assess practical weather risk for an outdoor activity in an Indian location.

    Uses the existing live forecast and deterministic thresholds, so this tool
    does not require another LLM call. It returns a simple LOW/MODERATE/HIGH
    risk level plus the factors that drove the decision.
    """
    weather = get_weather(location)
    if not weather.get("success"):
        return weather

    current = weather.get("current", {})
    daily = weather.get("daily", {})
    rain_probability = (daily.get("precipitation_probability_max") or [0])[0]
    precipitation = (daily.get("precipitation_sum") or [0])[0]
    wind = (daily.get("wind_speed_10m_max") or [0])[0]
    apparent_max = (daily.get("apparent_temperature_max") or [0])[0]
    weather_code = (daily.get("weather_code") or [0])[0]

    score = 0
    factors: list[str] = []

    if rain_probability >= 70:
        score += 2
        factors.append(f"high precipitation probability ({rain_probability}%)")
    elif rain_probability >= 40:
        score += 1
        factors.append(f"moderate precipitation probability ({rain_probability}%)")

    if precipitation >= 10:
        score += 2
        factors.append(f"heavy expected precipitation ({precipitation:.1f} mm)")
    elif precipitation >= 3:
        score += 1
        factors.append(f"expected precipitation ({precipitation:.1f} mm)")

    if wind >= 30:
        score += 2
        factors.append(f"strong wind ({wind:.1f} km/h)")
    elif wind >= 20:
        score += 1
        factors.append(f"elevated wind ({wind:.1f} km/h)")

    if apparent_max >= 40:
        score += 2
        factors.append(f"very high apparent temperature ({apparent_max:.1f}°C)")
    elif apparent_max >= 35:
        score += 1
        factors.append(f"high apparent temperature ({apparent_max:.1f}°C)")

    # Open-Meteo severe-weather codes: 95-99 represent thunderstorms.
    if weather_code >= 95:
        score += 2
        factors.append("thunderstorm risk in the forecast")

    if score >= 5:
        risk = "HIGH"
        recommendation = f"Avoid or postpone the {activity} if possible."
    elif score >= 2:
        risk = "MODERATE"
        recommendation = f"The {activity} is possible with precautions and a backup plan."
    else:
        risk = "LOW"
        recommendation = f"Conditions look generally suitable for the {activity}."

    return {
        "success": True,
        "location": weather["location"],
        "activity": activity,
        "risk_level": risk,
        "score": score,
        "recommendation": recommendation,
        "factors": factors or ["no major weather risk detected by the thresholds"],
        "data": {
            "rain_probability": rain_probability,
            "precipitation_mm": precipitation,
            "max_wind_kmh": wind,
            "apparent_temperature_max_c": apparent_max,
            "weather_code": weather_code,
            "current_temperature_c": current.get("temperature_2m"),
        },
    }


@mcp.tool()
def search_weather(query: str, top_k: int = 5) -> dict[str, Any]:
    """Search the Indian weather knowledge base using hybrid vector + BM25 retrieval."""
    if not query or not query.strip():
        raise ValueError("query cannot be empty")
    top_k = max(1, min(20, int(top_k)))
    documents = rag_service.retrieve_weather(query.strip(), top_k)
    return {
        "success": True,
        "query": query.strip(),
        "retrieval": "hybrid",
        "documents": documents,
        "count": len(documents),
    }


@mcp.tool()
def ask_weather(query: str, top_k: int = 5) -> dict[str, Any]:
    """Answer a natural-language Indian weather question using grounded hybrid RAG."""
    if not query or not query.strip():
        raise ValueError("query cannot be empty")
    top_k = max(1, min(20, int(top_k)))
    return rag_service.answer_weather_question(query=query.strip(), top_k=top_k)


@mcp.tool()
def sync_weather(locations: list[str]) -> dict[str, Any]:
    """Fetch and store fresh weather data for Indian locations."""
    if not locations:
        raise ValueError("locations cannot be empty")
    cleaned_locations = [location.strip() for location in locations if location and location.strip()]
    if not cleaned_locations:
        raise ValueError("No valid locations supplied")
    count = weather_client.sync_locations(cleaned_locations)
    return {"success": True, "locations": cleaned_locations, "documents_synced": count}


@mcp.tool()
def database_health() -> dict[str, Any]:
    """Check whether the configured PostgreSQL/Lakebase database is reachable."""
    try:
        connected = lakebase.check_connection()
        return {
            "success": connected,
            "backend": lakebase.DATABASE_BACKEND,
            "status": "ok" if connected else "unavailable",
        }
    except Exception as exc:
        return {
            "success": False,
            "backend": lakebase.DATABASE_BACKEND,
            "status": "error",
            "error": str(exc),
        }


if __name__ == "__main__":
    mcp.run()
