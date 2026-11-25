"""
Indian weather data client.

Uses Open-Meteo for free weather observations and forecasts.
Locations are resolved through OpenStreetMap Nominatim.

The normalized output is designed for the project's RAG pipeline
and PostgreSQL/Lakebase weather_documents table.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Dict, List, Optional, Tuple

import requests


import logging
import os
from threading import Lock
from urllib.parse import urlsplit

logger = logging.getLogger("weather-rag.weather-client")

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Nominatim's usage policy requires an identifying User-Agent; generic ones get blocked.
USER_AGENT = os.environ.get(
    "WEATHER_HTTP_USER_AGENT",
    "mcp-weather-intelligence/1.0 (+https://github.com/Atul1127/MCP-Enabled-Weather-Intelligence-System)",
)

REQUEST_TIMEOUT = float(os.environ.get("WEATHER_HTTP_TIMEOUT", "10"))
LOCATION_DELAY_SECONDS = 1

_GEOCODE_CACHE: Dict[str, Dict] = {}
_GEOCODE_CACHE_LOCK = Lock()
_GEOCODE_CACHE_MAX = 512


class UpstreamError(RuntimeError):
    """An external weather/geocoding provider failed or refused the request."""

    def __init__(self, service: str, status: Optional[int], message: str):
        super().__init__(f"{service}: {message}")
        self.service = service
        self.status = status


def _get_json(url: str, params: Dict, *, send_user_agent: bool = True):
    service = urlsplit(url).hostname or url
    headers = {"User-Agent": USER_AGENT} if send_user_agent else {}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    except requests.Timeout as exc:
        raise UpstreamError(service, None, f"timed out after {REQUEST_TIMEOUT:g}s") from exc
    except requests.RequestException as exc:
        raise UpstreamError(service, None, f"connection failed ({type(exc).__name__})") from exc
    if response.status_code >= 400:
        raise UpstreamError(service, response.status_code, f"HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError as exc:
        raise UpstreamError(service, response.status_code, "returned invalid JSON") from exc


# ---------------------------------------------------------------------------
# Location resolution
# ---------------------------------------------------------------------------

def _parse_coordinates(location: str) -> Optional[Dict]:
    """Accept direct coordinates such as "22.5726,88.3639"."""
    parts = [part.strip() for part in location.split(",")]
    if len(parts) != 2:
        return None
    try:
        latitude, longitude = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    if -90 <= latitude <= 90 and -180 <= longitude <= 180:
        return {"latitude": latitude, "longitude": longitude, "display_name": location, "state": None, "district": None}
    return None


def _geocode_nominatim(location: str) -> Optional[Dict]:
    results = _get_json(NOMINATIM_URL, {
        "q": f"{location}, India",
        "format": "json",
        "limit": 1,
        "countrycodes": "in",
        "addressdetails": 1,
    })
    if not results:
        return None
    result = results[0]
    address = result.get("address", {})
    return {
        "latitude": float(result["lat"]),
        "longitude": float(result["lon"]),
        "display_name": result.get("display_name", location),
        "state": address.get("state"),
        "district": address.get("state_district") or address.get("district") or address.get("county"),
    }


def _geocode_open_meteo(location: str) -> Optional[Dict]:
    # Open-Meteo matches a single place name, not "City, State" strings.
    name = location.split(",")[0].strip()
    data = _get_json(OPEN_METEO_GEOCODING_URL, {
        "name": name, "count": 1, "language": "en", "format": "json", "countryCode": "IN",
    })
    results = (data or {}).get("results") or []
    if not results:
        return None
    result = results[0]
    display = ", ".join(part for part in (result.get("name"), result.get("admin1"), result.get("country")) if part)
    return {
        "latitude": float(result["latitude"]),
        "longitude": float(result["longitude"]),
        "display_name": display or location,
        "state": result.get("admin1"),
        "district": result.get("admin2"),
    }


def geocode_location_details(location: str) -> Optional[Dict]:
    """Resolve an Indian location to coordinates and administrative metadata.

    Tries OpenStreetMap Nominatim first and falls back to Open-Meteo's geocoder,
    because Nominatim often blocks or rate-limits shared cloud IP addresses.
    Returns None when neither provider knows the place; raises UpstreamError
    when every provider failed.
    """
    location = location.strip()
    coordinates = _parse_coordinates(location)
    if coordinates:
        return coordinates

    key = location.lower()
    with _GEOCODE_CACHE_LOCK:
        cached = _GEOCODE_CACHE.get(key)
    if cached is not None:
        return dict(cached)

    errors: List[UpstreamError] = []
    result: Optional[Dict] = None
    for provider in (_geocode_nominatim, _geocode_open_meteo):
        try:
            result = provider(location)
        except UpstreamError as exc:
            logger.warning("Geocoding via %s failed: %s", exc.service, exc)
            errors.append(exc)
            continue
        if result:
            break

    if result is None:
        if errors and len(errors) == 2:
            raise errors[-1]
        return None

    with _GEOCODE_CACHE_LOCK:
        if len(_GEOCODE_CACHE) >= _GEOCODE_CACHE_MAX:
            _GEOCODE_CACHE.pop(next(iter(_GEOCODE_CACHE)))
        _GEOCODE_CACHE[key] = dict(result)
    return result


def geocode_location(location: str) -> Optional[Tuple[float, float, str]]:
    """Backward-compatible tuple form of geocode_location_details()."""
    details = geocode_location_details(location)
    if not details:
        return None
    return details["latitude"], details["longitude"], details["display_name"]


# ---------------------------------------------------------------------------
# Open-Meteo
# ---------------------------------------------------------------------------

def fetch_weather(latitude: float, longitude: float) -> Dict:
    """Fetch current weather, hourly data and a 7-day forecast (no API key needed)."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join([
            "temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation",
            "weather_code", "cloud_cover", "wind_speed_10m", "wind_direction_10m",
        ]),
        "hourly": ",".join([
            "temperature_2m", "precipitation_probability", "precipitation", "weather_code", "wind_speed_10m",
        ]),
        "daily": ",".join([
            "weather_code", "temperature_2m_max", "temperature_2m_min", "apparent_temperature_max",
            "apparent_temperature_min", "sunrise", "sunset", "precipitation_sum", "rain_sum",
            "precipitation_probability_max", "wind_speed_10m_max",
        ]),
        "timezone": "auto",
        "forecast_days": 7,
    }
    return _get_json(OPEN_METEO_URL, params)


# ---------------------------------------------------------------------------
# Weather code conversion
# ---------------------------------------------------------------------------

WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def weather_description(
    code: Optional[int],
) -> str:
    """
    Convert Open-Meteo WMO weather code to readable text.
    """

    if code is None:
        return "Unknown"

    return WEATHER_CODES.get(
        code,
        "Unknown weather condition",
    )


def weather_severity(
    code: Optional[int],
) -> str:
    """
    Classify weather conditions into application-level severity.

    This is NOT an official IMD warning classification.
    """

    if code is None:
        return "unknown"

    if code in {95, 96, 99}:
        return "severe"

    if code in {65, 67, 82, 86}:
        return "high"

    if code in {61, 63, 80, 81}:
        return "moderate"

    return "normal"


# ---------------------------------------------------------------------------
# Forecast hazard detection (shared by the HTTP API and the MCP server)
# ---------------------------------------------------------------------------

def _number(value, default: float = 0.0) -> float:
    """Open-Meteo returns null for missing values; treat them as 'no signal'."""
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def detect_alerts(daily: Dict) -> List[Dict]:
    """Deterministic application-level hazards from a daily Open-Meteo forecast.

    This is NOT an official IMD warning classification.
    """
    alerts: List[Dict] = []
    dates = daily.get("time") or []
    for index, date in enumerate(dates):
        probability = _number(_get_index(daily.get("precipitation_probability_max"), index))
        rain = _number(_get_index(daily.get("precipitation_sum"), index))
        wind = _number(_get_index(daily.get("wind_speed_10m_max"), index))
        apparent = _number(_get_index(daily.get("apparent_temperature_max"), index))
        code = int(_number(_get_index(daily.get("weather_code"), index)))

        if probability >= 70 and rain >= 15:
            alerts.append({"date": date, "severity": "HIGH", "hazard": "Heavy rain", "probability": probability, "details": f"{rain:.1f} mm expected with {probability:g}% precipitation probability.", "recommendation": "Avoid unnecessary outdoor activity and plan for travel disruption."})
        elif probability >= 70:
            alerts.append({"date": date, "severity": "MODERATE", "hazard": "High rain probability", "probability": probability, "details": f"Precipitation probability is {probability:g}%.", "recommendation": "Keep rain protection available and monitor the forecast."})
        if wind >= 40:
            alerts.append({"date": date, "severity": "HIGH", "hazard": "Strong wind", "details": f"Maximum forecast wind is {wind:.1f} km/h.", "recommendation": "Avoid exposed outdoor activities and secure loose objects."})
        if apparent >= 40:
            alerts.append({"date": date, "severity": "HIGH", "hazard": "Extreme heat", "details": f"Maximum apparent temperature is {apparent:.1f}°C.", "recommendation": "Limit strenuous outdoor activity, hydrate, and seek shade."})
        elif apparent >= 35:
            alerts.append({"date": date, "severity": "MODERATE", "hazard": "High heat", "details": f"Maximum apparent temperature is {apparent:.1f}°C.", "recommendation": "Take heat precautions during strenuous outdoor activity."})
        if code >= 95:
            alerts.append({"date": date, "severity": "HIGH", "hazard": "Thunderstorm", "details": f"Thunderstorm weather code {code} is forecast.", "recommendation": "Avoid exposed outdoor locations and seek sturdy shelter."})

    alerts.sort(key=lambda alert: (0 if alert["severity"] == "HIGH" else 1, str(alert["date"])))
    return alerts


def highest_severity(alerts: List[Dict]) -> str:
    if any(alert["severity"] == "HIGH" for alert in alerts):
        return "HIGH"
    return "MODERATE" if alerts else "NONE"


# ---------------------------------------------------------------------------
# Stable document IDs
# ---------------------------------------------------------------------------

def _make_id(
    prefix: str,
    raw: Dict,
) -> str:
    """
    Create a deterministic document ID.

    This allows repeated ingestion without creating duplicates.
    """

    canonical = json.dumps(
        {
            "prefix": prefix,
            **raw,
        },
        sort_keys=True,
        default=str,
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _get_index(
    values: Optional[List],
    index: int,
):
    """
    Safely retrieve a list value.
    """

    if not values or index >= len(values):
        return None

    return values[index]


# ---------------------------------------------------------------------------
# Current weather normalization
# ---------------------------------------------------------------------------

def normalize_current_weather(
    weather: Dict,
    location_label: str,
    location_details: Optional[Dict] = None,
) -> Dict:
    """
    Convert current Open-Meteo weather into a structured RAG document.
    """

    current = weather.get(
        "current",
        {},
    )

    weather_code = current.get(
        "weather_code"
    )

    latitude = (
        location_details.get("latitude")
        if location_details
        else None
    )

    longitude = (
        location_details.get("longitude")
        if location_details
        else None
    )

    state = (
        location_details.get("state")
        if location_details
        else None
    )

    district = (
        location_details.get("district")
        if location_details
        else None
    )

    current_time = current.get(
        "time"
    )

    forecast_date = (
        current_time[:10]
        if current_time
        else None
    )

    severity = weather_severity(
        weather_code
    )

    condition = weather_description(
        weather_code
    )

    headline = (
        f"Current weather in {location_label}"
    )

    narrative = (
        f"Current weather for {location_label}. "
        f"State: {state or 'Unknown'}. "
        f"District: {district or 'Unknown'}. "
        f"Temperature: "
        f"{current.get('temperature_2m')} °C. "
        f"Feels like: "
        f"{current.get('apparent_temperature')} °C. "
        f"Humidity: "
        f"{current.get('relative_humidity_2m')}%. "
        f"Condition: {condition}. "
        f"Severity: {severity}. "
        f"Precipitation: "
        f"{current.get('precipitation')} mm. "
        f"Cloud cover: "
        f"{current.get('cloud_cover')}%. "
        f"Wind speed: "
        f"{current.get('wind_speed_10m')} km/h. "
        f"Wind direction: "
        f"{current.get('wind_direction_10m')}°."
    )

    raw = {
        "location": location_label,
        "source": "open-meteo",
        "type": "current",
        "time": current_time,
    }

    return {
        "id": _make_id(
            "current",
            raw,
        ),
        "location": location_label,
        "state": state,
        "district": district,
        "latitude": latitude,
        "longitude": longitude,
        "source": "open-meteo",
        "source_type": "current",
        "headline": headline,
        "narrative_text": narrative,
        "forecast_date": forecast_date,
        "temperature_min_c": None,
        "temperature_max_c": current.get(
            "temperature_2m"
        ),
        "rainfall_mm": current.get(
            "precipitation"
        ),
        "precipitation_probability": None,
        "weather_code": weather_code,
        "severity": severity,
        "issued_at": current_time,
        "payload": weather,
        "synced_at": None,
    }


# ---------------------------------------------------------------------------
# Daily forecast normalization
# ---------------------------------------------------------------------------

def normalize_daily_forecasts(
    weather: Dict,
    location_label: str,
    location_details: Optional[Dict] = None,
) -> List[Dict]:
    """
    Convert daily Open-Meteo forecasts into structured RAG documents.
    """

    daily = weather.get(
        "daily",
        {},
    )

    dates = daily.get(
        "time",
        [],
    )

    documents = []

    latitude = (
        location_details.get("latitude")
        if location_details
        else None
    )

    longitude = (
        location_details.get("longitude")
        if location_details
        else None
    )

    state = (
        location_details.get("state")
        if location_details
        else None
    )

    district = (
        location_details.get("district")
        if location_details
        else None
    )

    for index, date in enumerate(dates):

        weather_code = _get_index(
            daily.get("weather_code"),
            index,
        )

        max_temp = _get_index(
            daily.get("temperature_2m_max"),
            index,
        )

        min_temp = _get_index(
            daily.get("temperature_2m_min"),
            index,
        )

        precipitation = _get_index(
            daily.get("precipitation_sum"),
            index,
        )

        rain = _get_index(
            daily.get("rain_sum"),
            index,
        )

        precipitation_probability = _get_index(
            daily.get(
                "precipitation_probability_max"
            ),
            index,
        )

        wind = _get_index(
            daily.get("wind_speed_10m_max"),
            index,
        )

        condition = weather_description(
            weather_code
        )

        severity = weather_severity(
            weather_code
        )

        headline = (
            f"Weather forecast for "
            f"{location_label} on {date}"
        )

        narrative = (
            f"Weather forecast for "
            f"{location_label} on {date}. "
            f"State: {state or 'Unknown'}. "
            f"District: {district or 'Unknown'}. "
            f"Condition: {condition}. "
            f"Severity: {severity}. "
            f"Minimum temperature: "
            f"{min_temp} °C. "
            f"Maximum temperature: "
            f"{max_temp} °C. "
            f"Total precipitation: "
            f"{precipitation} mm. "
            f"Rainfall: "
            f"{rain} mm. "
            f"Maximum precipitation probability: "
            f"{precipitation_probability}%. "
            f"Maximum wind speed: "
            f"{wind} km/h."
        )

        raw = {
            "location": location_label,
            "date": date,
            "type": "daily_forecast",
        }

        documents.append(
            {
                "id": _make_id(
                    "forecast",
                    raw,
                ),
                "location": location_label,
                "state": state,
                "district": district,
                "latitude": latitude,
                "longitude": longitude,
                "source": "open-meteo",
                "source_type": "forecast",
                "headline": headline,
                "narrative_text": narrative,
                "forecast_date": date,
                "temperature_min_c": min_temp,
                "temperature_max_c": max_temp,
                "rainfall_mm": rain,
                "precipitation_probability": (
                    precipitation_probability
                ),
                "weather_code": weather_code,
                "severity": severity,
                "issued_at": date,
                "payload": {
                    "date": date,
                    "weather_code": weather_code,
                    "condition": condition,
                    "temperature_min_c": min_temp,
                    "temperature_max_c": max_temp,
                    "precipitation_mm": precipitation,
                    "rain_mm": rain,
                    "precipitation_probability": (
                        precipitation_probability
                    ),
                    "max_wind_speed_kmh": wind,
                },
                "synced_at": None,
            }
        )

    return documents


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def upsert_documents(
    documents: List[Dict],
) -> int:
    """
    Upsert normalized weather documents into PostgreSQL/Lakebase.

    Lakebase is imported lazily so weather API functionality can be
    tested locally without Databricks credentials.
    """

    import lakebase

    if not documents:
        return 0

    sql = """
        INSERT INTO weather_documents (
            id,
            location,
            state,
            district,
            latitude,
            longitude,
            source,
            source_type,
            headline,
            narrative_text,
            forecast_date,
            temperature_min_c,
            temperature_max_c,
            rainfall_mm,
            precipitation_probability,
            weather_code,
            severity,
            issued_at,
            payload,
            synced_at
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            now()
        )
        ON CONFLICT (id)
        DO UPDATE SET
            location = EXCLUDED.location,
            state = EXCLUDED.state,
            district = EXCLUDED.district,
            latitude = EXCLUDED.latitude,
            longitude = EXCLUDED.longitude,
            source = EXCLUDED.source,
            source_type = EXCLUDED.source_type,
            headline = EXCLUDED.headline,
            narrative_text = EXCLUDED.narrative_text,
            forecast_date = EXCLUDED.forecast_date,
            temperature_min_c = EXCLUDED.temperature_min_c,
            temperature_max_c = EXCLUDED.temperature_max_c,
            rainfall_mm = EXCLUDED.rainfall_mm,
            precipitation_probability =
                EXCLUDED.precipitation_probability,
            weather_code = EXCLUDED.weather_code,
            severity = EXCLUDED.severity,
            issued_at = EXCLUDED.issued_at,
            payload = EXCLUDED.payload,
            synced_at = now()
    """

    params = []

    for document in documents:

        params.append(
            (
                document.get("id"),
                document.get("location"),
                document.get("state"),
                document.get("district"),
                document.get("latitude"),
                document.get("longitude"),
                document.get(
                    "source",
                    "open-meteo",
                ),
                document.get("source_type"),
                document.get("headline"),
                document.get("narrative_text"),
                document.get("forecast_date"),
                document.get("temperature_min_c"),
                document.get("temperature_max_c"),
                document.get("rainfall_mm"),
                document.get(
                    "precipitation_probability"
                ),
                document.get("weather_code"),
                document.get("severity"),
                document.get("issued_at"),
                json.dumps(
                    document.get("payload") or {},
                    default=str,
                ),
            )
        )

    with lakebase.get_connection() as connection:

        with connection.cursor() as cursor:

            cursor.executemany(
                sql,
                params,
            )

            connection.commit()

            return cursor.rowcount


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------

def sync_locations(
    locations: List[str],
    limit: int = 50,
) -> int:
    """
    Fetch and store weather data for Indian locations.

    `limit` is retained for backward compatibility with the
    previous NWS implementation.
    """

    del limit

    total = 0

    for location in locations:

        try:

            location_details = (
                geocode_location_details(
                    location
                )
            )

            if not location_details:

                print(
                    f"Could not resolve location: "
                    f"{location}"
                )

                continue

            latitude = location_details[
                "latitude"
            ]

            longitude = location_details[
                "longitude"
            ]

            label = location_details[
                "display_name"
            ]

            print(
                f"Fetching weather for {label} "
                f"({latitude}, {longitude})"
            )

            weather = fetch_weather(
                latitude,
                longitude,
            )

            documents = []

            documents.append(
                normalize_current_weather(
                    weather,
                    label,
                    location_details,
                )
            )

            documents.extend(
                normalize_daily_forecasts(
                    weather,
                    label,
                    location_details,
                )
            )

            count = upsert_documents(
                documents
            )

            total += count

            print(
                f"Synced {count} documents "
                f"for {label}"
            )

        except requests.RequestException as exc:

            print(
                f"Weather API error for "
                f"{location}: {exc}"
            )

        except Exception as exc:

            print(
                f"Unexpected error for "
                f"{location}: {exc}"
            )

        # Respect Nominatim/Open-Meteo usage.
        time.sleep(
            LOCATION_DELAY_SECONDS
        )

    return total
