"""Indian Weather RAG API and intelligence dashboard."""

from __future__ import annotations

import logging
import os

from flask import Flask, jsonify, render_template, request

import lakebase
import rag_service
import weather_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-rag")
app = Flask(__name__)


@app.route("/", methods=["GET"])
def dashboard():
    return render_template("dashboard.html")


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok", "service": "indian-weather-rag"})


@app.route("/weather/current", methods=["POST"])
def weather_current():
    body = request.get_json(silent=True) or {}
    location = body.get("location")
    if not isinstance(location, str) or not location.strip():
        return jsonify({"error": "Missing or invalid 'location'"}), 400
    try:
        details = weather_client.geocode_location_details(location.strip())
        if not details:
            return jsonify({"error": f"Could not resolve location: {location.strip()}"}), 404
        weather = weather_client.fetch_weather(details["latitude"], details["longitude"])
        return jsonify({"success": True, "location": details, "current": weather.get("current", {}), "daily": weather.get("daily", {})})
    except Exception as exc:
        logger.exception("Current weather request failed")
        return jsonify({"error": "Failed to fetch weather", "details": str(exc)}), 500


@app.route("/weather/alerts", methods=["POST"])
def weather_alerts():
    body = request.get_json(silent=True) or {}
    location = body.get("location")
    if not isinstance(location, str) or not location.strip():
        return jsonify({"error": "Missing or invalid 'location'"}), 400
    try:
        details = weather_client.geocode_location_details(location.strip())
        if not details:
            return jsonify({"error": f"Could not resolve location: {location.strip()}"}), 404
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
    except Exception as exc:
        logger.exception("Weather alert request failed")
        return jsonify({"error": "Failed to generate alerts", "details": str(exc)}), 500


@app.route("/weather/ask", methods=["POST"])
def weather_ask():
    body = request.get_json(silent=True) or {}
    query = body.get("query")
    if not query or not isinstance(query, str):
        return jsonify({"error": "Missing or invalid 'query' in request body"}), 400
    query = query.strip()
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400
    try:
        top_k = int(body.get("top_k", 5))
    except (TypeError, ValueError):
        return jsonify({"error": "'top_k' must be an integer"}), 400
    top_k = max(1, min(20, top_k))
    try:
        return jsonify(rag_service.answer_weather_question(query=query, top_k=top_k))
    except Exception as exc:
        logger.exception("Weather RAG request failed")
        return jsonify({"error": "Failed to generate weather answer", "details": str(exc)}), 500


@app.route("/weather/sync", methods=["POST"])
def weather_sync():
    body = request.get_json(silent=True) or {}
    locations = body.get("locations")
    if not isinstance(locations, list) or not locations:
        return jsonify({"error": "Missing or invalid 'locations' list"}), 400
    cleaned_locations = [location.strip() for location in locations if isinstance(location, str) and location.strip()]
    if not cleaned_locations:
        return jsonify({"error": "No valid locations were provided"}), 400
    try:
        lakebase.ensure_weather_tables(embedding_dim=384)
        synced = weather_client.sync_locations(cleaned_locations)
        return jsonify({"status": "success", "synced": synced, "locations": cleaned_locations})
    except Exception as exc:
        logger.exception("Weather synchronization failed")
        return jsonify({"error": "Weather synchronization failed", "details": str(exc)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(Exception)
def handle_exception(error):
    logger.exception("Unhandled application error")
    return jsonify({"error": "Internal server error", "details": str(error)}), 500


if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", "8000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host=host, port=port)
