"""Indian Weather RAG API and intelligence dashboard."""
from __future__ import annotations

import asyncio
import logging
import os
import secrets

from flask import Flask, jsonify, render_template, request

import lakebase
import rag_service
import weather_client
from weather_agent_core import WeatherAgent
from weather_agent_core.security import inspect_text, validate_location, validate_top_k, validate_user_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-rag")
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("WEATHER_MAX_REQUEST_BYTES", "262144"))


def _safe_location(value: object) -> str:
    return validate_location(value)  # type: ignore[arg-type]


def _sync_authorized() -> bool:
    """Keep the write-heavy sync endpoint disabled unless explicitly enabled."""
    if os.environ.get("WEATHER_ALLOW_SYNC", "0") != "1":
        return False
    configured = os.environ.get("WEATHER_ADMIN_API_KEY")
    supplied = request.headers.get("X-Weather-Admin-Key", "")
    return bool(configured) and secrets.compare_digest(supplied, configured)


@app.route("/", methods=["GET"])
def dashboard():
    return render_template("dashboard.html")


@app.route("/healthz", methods=["GET"])
def healthz():
    """Liveness probe: only confirms that the HTTP process is running."""
    return jsonify({"status": "ok", "service": "indian-weather-rag"})


@app.route("/readyz", methods=["GET"])
def readyz():
    """Readiness probe for dependencies required by the main API path.

    External APIs are intentionally not probed here because readiness should
    remain fast and deterministic; transient provider failures are handled by
    the request-level retry/error paths.
    """
    checks = {
        "gemini_api_key": bool(os.environ.get("GEMINI_API_KEY")),
        "rag_store": False,
    }
    try:
        rag_service.get_rag_pipeline()
        checks["rag_store"] = True
    except Exception:
        logger.exception("RAG readiness check failed")

    ready = all(checks.values())
    return jsonify({"status": "ready" if ready else "not_ready", "checks": checks}), 200 if ready else 503


@app.route("/weather/current", methods=["POST"])
def weather_current():
    body = request.get_json(silent=True) or {}
    try:
        location = _safe_location(body.get("location"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        details = weather_client.geocode_location_details(location)
        if not details:
            return jsonify({"error": "Location could not be resolved"}), 404
        weather = weather_client.fetch_weather(details["latitude"], details["longitude"])
        return jsonify({"success": True, "location": details, "current": weather.get("current", {}), "daily": weather.get("daily", {})})
    except Exception:
        logger.exception("Current weather request failed")
        return jsonify({"error": "Failed to fetch weather"}), 502


@app.route("/weather/alerts", methods=["POST"])
def weather_alerts():
    body = request.get_json(silent=True) or {}
    try:
        location = _safe_location(body.get("location"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        details = weather_client.geocode_location_details(location)
        if not details:
            return jsonify({"error": "Location could not be resolved"}), 404
        weather = weather_client.fetch_weather(details["latitude"], details["longitude"])
        daily = weather.get("daily", {})
        alerts = []
        dates = daily.get("time", [])
        for index, date in enumerate(dates):
            probability = (daily.get("precipitation_probability_max") or [0] * len(dates))[index]
            rain = (daily.get("precipitation_sum") or [0] * len(dates))[index]
            wind = (daily.get("wind_speed_10m_max") or [0] * len(dates))[index]
            apparent = (daily.get("apparent_temperature_max") or [0] * len(dates))[index]
            code = (daily.get("weather_code") or [0] * len(dates))[index]
            if probability >= 70 and rain >= 15:
                alerts.append({"date": date, "severity": "HIGH", "hazard": "Heavy rain", "details": f"{rain:.1f} mm expected with {probability}% probability.", "recommendation": "Avoid unnecessary outdoor activity and plan for travel disruption."})
            elif probability >= 70:
                alerts.append({"date": date, "severity": "MODERATE", "hazard": "High rain probability", "details": f"Precipitation probability is {probability}%.", "recommendation": "Keep rain protection available and monitor the forecast."})
            if wind >= 40:
                alerts.append({"date": date, "severity": "HIGH", "hazard": "Strong wind", "details": f"Maximum forecast wind is {wind:.1f} km/h.", "recommendation": "Avoid exposed outdoor activities and secure loose objects."})
            if apparent >= 40:
                alerts.append({"date": date, "severity": "HIGH", "hazard": "Extreme heat", "details": f"Maximum apparent temperature is {apparent:.1f}°C.", "recommendation": "Limit strenuous outdoor activity, hydrate, and seek shade."})
            elif apparent >= 35:
                alerts.append({"date": date, "severity": "MODERATE", "hazard": "High heat", "details": f"Maximum apparent temperature is {apparent:.1f}°C.", "recommendation": "Take heat precautions during strenuous outdoor activity."})
            if code >= 95:
                alerts.append({"date": date, "severity": "HIGH", "hazard": "Thunderstorm", "details": f"Thunderstorm weather code {code} is forecast.", "recommendation": "Avoid exposed outdoor locations and seek sturdy shelter."})
        return jsonify({"success": True, "location": details, "alerts": alerts, "alert_count": len(alerts), "highest_severity": "HIGH" if any(a["severity"] == "HIGH" for a in alerts) else ("MODERATE" if alerts else "NONE")})
    except Exception:
        logger.exception("Weather alert request failed")
        return jsonify({"error": "Failed to generate alerts"}), 502


@app.route("/weather/ask", methods=["POST"])
def weather_ask():
    """RAG-only knowledge endpoint; the full agent is exposed separately."""
    body = request.get_json(silent=True) or {}
    query = body.get("query")
    if not query or not isinstance(query, str):
        return jsonify({"error": "Missing or invalid 'query' in request body"}), 400
    query = query.strip()
    if not query:
        return jsonify({"error": "Missing or invalid 'query' in request body"}), 400
    security = inspect_text(query)
    if security["suspicious"]:
        return jsonify({"error": "Query contains a blocked prompt-injection signal"}), 400
    try:
        query = validate_user_query(query)
        top_k = validate_top_k(body.get("top_k", 5))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        return jsonify(rag_service.answer_weather_question(query=query, top_k=top_k))
    except Exception:
        logger.exception("Weather RAG request failed")
        return jsonify({"error": "Failed to generate weather answer"}), 502


@app.route("/weather/agent", methods=["POST"])
def weather_agent():
    """Run the canonical LangGraph + MCP + RAG WeatherAgent."""
    body = request.get_json(silent=True) or {}
    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "Missing or invalid 'query' in request body"}), 400
    try:
        query = validate_user_query(query)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        result = asyncio.run(WeatherAgent().run(query))
        return jsonify(result), 200 if result.get("success") else 502
    except Exception:
        logger.exception("Weather agent request failed")
        return jsonify({"error": "Failed to generate agent answer"}), 502


@app.route("/weather/sync", methods=["POST"])
def weather_sync():
    if not _sync_authorized():
        return jsonify({"error": "Weather synchronization is disabled or unauthorized"}), 403
    body = request.get_json(silent=True) or {}
    locations = body.get("locations")
    if not isinstance(locations, list) or not locations:
        return jsonify({"error": "Missing or invalid 'locations' list"}), 400
    if len(locations) > 50:
        return jsonify({"error": "A maximum of 50 locations is allowed"}), 400
    try:
        cleaned_locations = [_safe_location(location) for location in locations]
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        lakebase.ensure_weather_tables(embedding_dim=384)
        synced = weather_client.sync_locations(cleaned_locations)
        return jsonify({"status": "success", "synced": synced, "locations": cleaned_locations})
    except Exception:
        logger.exception("Weather synchronization failed")
        return jsonify({"error": "Weather synchronization failed"}), 502


@app.errorhandler(413)
def request_too_large(error):
    return jsonify({"error": "Request body is too large"}), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(Exception)
def handle_exception(error):
    logger.exception("Unhandled application error")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", "8000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host=host, port=port)
