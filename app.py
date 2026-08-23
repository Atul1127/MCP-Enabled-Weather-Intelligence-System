"""Indian Weather RAG API.

Provides:
    GET  /healthz
    POST /weather/ask
    POST /weather/sync

The MCP server is the agent/tool orchestration layer; this Flask app remains
as the existing HTTP/RAG interface.

No paid LLM API is required.
"""

from __future__ import annotations

import logging
import os

from flask import Flask, jsonify, request

import lakebase
import rag_service
import weather_client


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-rag")
app = Flask(__name__)


@app.route("/healthz", methods=["GET"])
def healthz():
    """Basic application health check without database access."""
    return jsonify({"status": "ok", "service": "indian-weather-rag"})


@app.route("/weather/ask", methods=["POST"])
def weather_ask():
    """Ask a weather question using the hybrid RAG pipeline."""
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
    """Fetch and store weather data for Indian locations."""
    body = request.get_json(silent=True) or {}
    locations = body.get("locations")

    if not isinstance(locations, list) or not locations:
        return jsonify({"error": "Missing or invalid 'locations' list"}), 400

    cleaned_locations = [
        location.strip()
        for location in locations
        if isinstance(location, str) and location.strip()
    ]

    if not cleaned_locations:
        return jsonify({"error": "No valid locations were provided"}), 400

    try:
        lakebase.ensure_weather_tables(embedding_dim=384)
        synced = weather_client.sync_locations(cleaned_locations)
        return jsonify({
            "status": "success",
            "synced": synced,
            "locations": cleaned_locations,
        })
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
