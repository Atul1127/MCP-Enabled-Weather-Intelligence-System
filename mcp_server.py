"""Indian Weather Intelligence MCP server with MCP SDK v1/v2 compatibility."""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from threading import Lock
from typing import Any
import copy
import json
import os

import lakebase, weather_client

try:
    from mcp.server import MCPServer
except ImportError:
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError:
        from mcp.server.fastmcp import FastMCP as MCPServer

from observability import emit, span

mcp = MCPServer("indian-weather-intelligence")

# search_weather and ask_weather are separate MCP capabilities, but both
# intentionally consume the same retrieval result for an identical request.
# This prevents running dense retrieval, BM25 fusion, reranking and context
# compression twice during a knowledge plan that requires both capabilities.
_RAG_CACHE_MAX = max(1, int(os.environ.get("WEATHER_RAG_CACHE_SIZE", "32")))
_RAG_CACHE: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
_RAG_CACHE_LOCK = Lock()
_RAG_PIPELINE: Any = None
_RAG_PIPELINE_LOCK = Lock()


def _rag_pipeline() -> Any:
    """Return one process-local RAG pipeline instance."""
    global _RAG_PIPELINE
    if _RAG_PIPELINE is None:
        with _RAG_PIPELINE_LOCK:
            if _RAG_PIPELINE is None:
                from rag.pipeline import RAGPipeline
                _RAG_PIPELINE = RAGPipeline()
    return _RAG_PIPELINE


def _rag_cache_key(
    query: str,
    top_k: int,
    location: str | None,
    state: str | None,
) -> tuple[Any, ...]:
    return (query, int(top_k), location, state)


def _retrieve_weather_knowledge(
    query: str,
    top_k: int,
    location: str | None,
    state: str | None,
) -> dict[str, Any]:
    """Retrieve once and reuse identical grounded evidence."""
    key = _rag_cache_key(query, top_k, location, state)
    with _RAG_CACHE_LOCK:
        cached = _RAG_CACHE.get(key)
        if cached is not None:
            _RAG_CACHE.move_to_end(key)
            return copy.deepcopy(cached)

        result = _rag_pipeline().retrieve(
            query,
            location=location,
            state=state,
            top_k=top_k,
        )
        payload = {
            "success": True,
            "query": query,
            "intent": result.plan.intent,
            "documents": result.documents,
            "context": result.context,
            "sources": result.sources,
        }
        _RAG_CACHE[key] = copy.deepcopy(payload)
        _RAG_CACHE.move_to_end(key)
        while len(_RAG_CACHE) > _RAG_CACHE_MAX:
            _RAG_CACHE.popitem(last=False)
        return copy.deepcopy(payload)


@mcp.resource("weather://capabilities")
def weather_capabilities() -> str:
    """Describe the server's weather capabilities and when to use them."""
    return json.dumps({
        "tools": ["get_weather", "get_forecast", "get_weather_alerts", "assess_weather_risk", "search_weather", "ask_weather", "sync_weather", "database_health"],
        "resources": ["weather://capabilities", "weather://policy", "weather://forecast/{location}/{date}"],
        "prompts": ["weather_analysis", "compare_weather", "activity_risk"],
        "principles": {
            "live_data": "Use weather tools for current or forecast observations.",
            "knowledge": "Use search_weather/ask_weather for grounded weather knowledge.",
            "resources": "Use resources for read-only contextual data.",
            "prompts": "Use prompts as reusable interaction templates; generation remains in the host agent."
        }
    }, ensure_ascii=False)

@mcp.resource("weather://policy")
def weather_policy() -> str:
    """Expose safety and grounding policy for host agents."""
    return json.dumps({
        "grounding": ["Never invent live weather values.", "Prefer MCP observations for current and forecast facts.", "Use RAG evidence for static weather knowledge."],
        "safety": ["Risk assessments are advisory, not official warnings.", "Official government alerts take precedence over application-level hazard detection."],
        "citations": ["Preserve source identifiers returned by MCP and RAG.", "Do not present unsupported claims as observed facts."]
    }, ensure_ascii=False)

@mcp.resource("weather://forecast/{location}/{date}")
def weather_forecast_resource(location: str, date: str = "tomorrow") -> str:
    """Read a forecast snapshot as a resource without invoking a model tool."""
    details = weather_client.geocode_location_details(location.strip())
    if not details:
        return json.dumps({"success": False, "error": f"Could not resolve location: {location}"})
    weather = weather_client.fetch_weather(details["latitude"], details["longitude"])
    target = _target_date(date, weather.get("daily", {}))
    return json.dumps({"success": True, "location": details, "forecast": _daily_summary(weather, target), "source": "open-meteo"}, ensure_ascii=False)

@mcp.prompt()
def weather_analysis(query: str, evidence: str = "") -> str:
    """Create a grounded weather-analysis instruction for the host agent."""
    return f"Analyze this weather question using only the supplied evidence. Separate live observations from general knowledge, state uncertainty, and avoid inventing values.\n\nQUESTION:\n{query}\n\nEVIDENCE:\n{evidence}"

@mcp.prompt()
def compare_weather(location_a: str, location_b: str, date: str = "tomorrow") -> str:
    """Create a structured comparison prompt for two locations."""
    return f"Compare {location_a} and {location_b} for {date}. Compare temperature, precipitation, wind, hazards, and practical implications. Use only retrieved MCP/RAG evidence and cite sources."

@mcp.prompt()
def activity_risk(location: str, activity: str, date: str = "tomorrow") -> str:
    """Create a reusable activity-risk analysis prompt."""
    return f"Assess the weather risk for {activity} in {location} on {date}. Use live forecast evidence, identify hazards, give a cautious recommendation, and distinguish advisory assessment from official warnings."

def _target_date(value: str | None, daily: dict[str, Any]) -> str:
    dates = daily.get("time") or []
    if not dates:
        raise ValueError("Forecast contains no dates")
    text = (value or "tomorrow").strip().lower()
    if text == "today": return dates[0]
    if text == "tomorrow": return dates[1] if len(dates) > 1 else dates[0]
    try: return datetime.strptime(text, "%Y-%m-%d").date().isoformat()
    except ValueError as exc: raise ValueError("date must be 'today', 'tomorrow', or YYYY-MM-DD") from exc

def _daily_summary(weather: dict[str, Any], target: str) -> dict[str, Any]:
    daily = weather.get("daily", {}); dates = daily.get("time") or []
    if target not in dates: raise ValueError(f"No forecast available for {target}")
    i = dates.index(target)
    def value(name: str, default: Any = None):
        values = daily.get(name) or []; return values[i] if i < len(values) else default
    code = value("weather_code")
    return {"date": target, "condition": weather_client.weather_description(code), "weather_code": code, "severity": weather_client.weather_severity(code), "temperature_min_c": value("temperature_2m_min"), "temperature_max_c": value("temperature_2m_max"), "apparent_temperature_max_c": value("apparent_temperature_max"), "precipitation_mm": value("precipitation_sum"), "rain_mm": value("rain_sum"), "precipitation_probability_pct": value("precipitation_probability_max"), "max_wind_kmh": value("wind_speed_10m_max"), "sunrise": value("sunrise"), "sunset": value("sunset")}

def _current_time_of_day(current_time: str | None, daily: dict[str, Any]) -> str:
    if not current_time: return "unknown"
    try:
        current = datetime.fromisoformat(current_time); date = current_time[:10]; dates = daily.get("time") or []
        i = dates.index(date) if date in dates else 0; sunrise = (daily.get("sunrise") or [None])[i]; sunset = (daily.get("sunset") or [None])[i]
        if sunrise and sunset: return "day" if sunrise <= current_time <= sunset else "night"
    except (ValueError, IndexError, TypeError): pass
    return "unknown"

def _current_summary(current: dict[str, Any], daily: dict[str, Any], timezone: str | None) -> dict[str, Any]:
    code = current.get("weather_code")
    return {"observation_time": current.get("time"), "timezone": timezone, "time_of_day": _current_time_of_day(current.get("time"), daily), "condition": weather_client.weather_description(code), "weather_code": code, "temperature_c": current.get("temperature_2m"), "apparent_temperature_c": current.get("apparent_temperature"), "relative_humidity_pct": current.get("relative_humidity_2m"), "cloud_cover_pct": current.get("cloud_cover"), "precipitation_mm": current.get("precipitation"), "wind_speed_kmh": current.get("wind_speed_10m"), "wind_direction_deg": current.get("wind_direction_10m")}

@mcp.tool()
def get_weather(location: str) -> dict[str, Any]:
    """Get current weather and a 7-day forecast."""
    location = location.strip() if location else ""
    if not location: raise ValueError("location cannot be empty")
    details = weather_client.geocode_location_details(location)
    if not details: return {"success": False, "error": f"Could not resolve location: {location}"}
    weather = weather_client.fetch_weather(details["latitude"], details["longitude"]); current = weather.get("current", {}); daily = weather.get("daily", {})
    return {"success": True, "location": details, "timezone": weather.get("timezone"), "current": current, "current_summary": _current_summary(current, daily, weather.get("timezone")), "daily": daily, "forecast_summary": [_daily_summary(weather, d) for d in daily.get("time") or []]}

@mcp.tool()
def get_forecast(location: str, date: str = "tomorrow") -> dict[str, Any]:
    """Get a forecast for today, tomorrow, or YYYY-MM-DD."""
    location = location.strip() if location else ""
    if not location: raise ValueError("location cannot be empty")
    details = weather_client.geocode_location_details(location)
    if not details: return {"success": False, "error": f"Could not resolve location: {location}"}
    weather = weather_client.fetch_weather(details["latitude"], details["longitude"]); target = _target_date(date, weather.get("daily", {}))
    return {"success": True, "location": details, "forecast": _daily_summary(weather, target), "source": "open-meteo"}

@mcp.tool()
def get_weather_alerts(location: str) -> dict[str, Any]:
    """Detect actionable hazards from the live 7-day forecast."""
    weather = get_weather(location)
    if not weather.get("success"): return weather
    daily = weather.get("daily", {}); alerts = []
    for i, target in enumerate(daily.get("time", [])):
        probabilities = daily.get("precipitation_probability_max") or []; rains = daily.get("precipitation_sum") or []; winds = daily.get("wind_speed_10m_max") or []; apparents = daily.get("apparent_temperature_max") or []; codes = daily.get("weather_code") or []
        probability = probabilities[i] if i < len(probabilities) else 0; rain = rains[i] if i < len(rains) else 0; wind = winds[i] if i < len(winds) else 0; apparent = apparents[i] if i < len(apparents) else 0; code = codes[i] if i < len(codes) else 0
        if probability >= 70 and rain >= 15: alerts.append({"date": target, "severity": "HIGH", "hazard": "Heavy rain", "probability": probability, "details": f"{rain:.1f} mm expected with {probability}% precipitation probability."})
        elif probability >= 70: alerts.append({"date": target, "severity": "MODERATE", "hazard": "High rain probability", "probability": probability, "details": f"Precipitation probability is {probability}%."})
        if wind >= 40: alerts.append({"date": target, "severity": "HIGH", "hazard": "Strong wind", "details": f"Maximum forecast wind is {wind:.1f} km/h."})
        if apparent >= 40: alerts.append({"date": target, "severity": "HIGH", "hazard": "Extreme heat", "details": f"Maximum apparent temperature is {apparent:.1f}°C."})
        elif apparent >= 35: alerts.append({"date": target, "severity": "MODERATE", "hazard": "High heat", "details": f"Maximum apparent temperature is {apparent:.1f}°C."})
        if code >= 95: alerts.append({"date": target, "severity": "HIGH", "hazard": "Thunderstorm", "details": f"Forecast weather code is {code}."})
    alerts.sort(key=lambda x: (0 if x["severity"] == "HIGH" else 1, x["date"]))
    return {"success": True, "location": weather["location"], "alerts": alerts, "alert_count": len(alerts), "highest_severity": alerts[0]["severity"] if alerts else "NONE", "disclaimer": "Application-level forecast hazard detection; not an official government warning."}

@mcp.tool()
def assess_weather_risk(location: str, activity: str = "outdoor activity", date: str = "tomorrow") -> dict[str, Any]:
    """Assess activity risk for a specific forecast date."""
    activity = activity.strip() if activity else "outdoor activity"; activity = activity or "outdoor activity"; weather = get_weather(location)
    if not weather.get("success"): return weather
    target = _target_date(date, weather.get("daily", {})); forecast = _daily_summary(weather, target); rain_probability = forecast["precipitation_probability_pct"] or 0; precipitation = forecast["precipitation_mm"] or 0; wind = forecast["max_wind_kmh"] or 0; apparent = forecast["apparent_temperature_max_c"] or 0; code = forecast["weather_code"] or 0; score = 0; factors = []
    if rain_probability >= 70: score += 2; factors.append(f"high precipitation probability ({rain_probability}%)")
    elif rain_probability >= 40: score += 1; factors.append(f"moderate precipitation probability ({rain_probability}%)")
    if precipitation >= 10: score += 2; factors.append(f"heavy expected precipitation ({precipitation:.1f} mm)")
    elif precipitation >= 3: score += 1; factors.append(f"expected precipitation ({precipitation:.1f} mm)")
    if wind >= 30: score += 2; factors.append(f"strong wind ({wind:.1f} km/h)")
    elif wind >= 20: score += 1; factors.append(f"elevated wind ({wind:.1f} km/h)")
    if apparent >= 40: score += 2; factors.append(f"very high apparent temperature ({apparent:.1f}°C)")
    elif apparent >= 35: score += 1; factors.append(f"high apparent temperature ({apparent:.1f}°C)")
    if code >= 95: score += 2; factors.append("thunderstorm risk in the forecast")
    risk = "HIGH" if score >= 5 else "MODERATE" if score >= 2 else "LOW"; recommendation = f"Avoid or postpone the {activity} if possible." if risk == "HIGH" else f"The {activity} is possible with precautions and a backup plan." if risk == "MODERATE" else f"Conditions look generally suitable for the {activity}."
    return {"success": True, "location": weather["location"], "activity": activity, "date": target, "forecast": forecast, "risk_level": risk, "score": score, "recommendation": recommendation, "factors": factors or ["no major weather risk detected"]}

@mcp.tool()
def search_weather(query: str, top_k: int = 5, location: str | None = None, state: str | None = None) -> dict[str, Any]:
    """Search weather knowledge through the modular RAG pipeline."""
    query = query.strip() if query else ""
    if not query: return {"success": False, "error": "query cannot be empty"}
    with span("mcp.search_weather", trace_id="unknown", tool="search_weather") as info:
        payload = _retrieve_weather_knowledge(query, top_k, location, state)
        info["success"] = True
        info["documents"] = len(payload["documents"])
        info["sources"] = len(payload["sources"])
        emit("mcp.tool.result", trace_id="unknown", tool="search_weather", success=True, documents=len(payload["documents"]), sources=len(payload["sources"]))
        return payload

@mcp.tool()
def ask_weather(query: str, top_k: int = 5, location: str | None = None, state: str | None = None) -> dict[str, Any]:
    """Retrieve the same grounded weather knowledge evidence; generation is handled by the agent synthesizer."""
    query = query.strip() if query else ""
    if not query: return {"success": False, "error": "query cannot be empty"}
    with span("mcp.ask_weather", trace_id="unknown", tool="ask_weather") as info:
        payload = _retrieve_weather_knowledge(query, top_k, location, state)
        info["success"] = True
        info["documents"] = len(payload["documents"])
        info["sources"] = len(payload["sources"])
        emit("mcp.tool.result", trace_id="unknown", tool="ask_weather", success=True, documents=len(payload["documents"]), sources=len(payload["sources"]))
        return payload

@mcp.tool()
def sync_weather(locations: list[str]) -> dict[str, Any]:
    """Fetch and store fresh weather data for Indian locations."""
    cleaned = [x.strip() for x in locations if x and x.strip()] if locations else []
    if not cleaned: raise ValueError("locations cannot be empty")
    return {"success": True, "locations": cleaned, "documents_synced": weather_client.sync_locations(cleaned)}

@mcp.tool()
def database_health() -> dict[str, Any]:
    """Check PostgreSQL/Lakebase reachability."""
    try:
        connected = lakebase.check_connection(); return {"success": connected, "backend": lakebase.DATABASE_BACKEND, "status": "ok" if connected else "unavailable"}
    except Exception as exc: return {"success": False, "backend": lakebase.DATABASE_BACKEND, "status": "error", "error": str(exc)}

if __name__ == "__main__": mcp.run()
