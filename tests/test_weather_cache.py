"""Caching and rate-limit behaviour of the weather client."""
import weather_client


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_geocoding_is_cached_and_case_insensitive(monkeypatch):
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(params["q"])
        return _FakeResponse([{"lat": "19.07", "lon": "72.87", "display_name": "Mumbai", "address": {"state": "Maharashtra"}}])

    monkeypatch.setattr(weather_client.requests, "get", fake_get)
    monkeypatch.setattr(weather_client, "NOMINATIM_MIN_INTERVAL_SECONDS", 0.0)

    first = weather_client.geocode_location_details("Mumbai")
    second = weather_client.geocode_location_details("  mumbai ")
    assert first == second
    assert len(calls) == 1
    # Callers may mutate results without corrupting the cache.
    first["latitude"] = 0
    assert weather_client.geocode_location_details("Mumbai")["latitude"] == 19.07


def test_geocoding_misses_are_cached(monkeypatch):
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(1)
        return _FakeResponse([])

    monkeypatch.setattr(weather_client.requests, "get", fake_get)
    monkeypatch.setattr(weather_client, "NOMINATIM_MIN_INTERVAL_SECONDS", 0.0)
    assert weather_client.geocode_location_details("Nowhereville") is None
    assert weather_client.geocode_location_details("Nowhereville") is None
    assert len(calls) == 1


def test_weather_is_cached_per_location(monkeypatch):
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append((params["latitude"], params["longitude"]))
        return _FakeResponse({"current": {"temperature_2m": 30}})

    monkeypatch.setattr(weather_client.requests, "get", fake_get)
    weather_client.fetch_weather(19.07, 72.87)
    weather_client.fetch_weather(19.07, 72.87)
    weather_client.fetch_weather(28.61, 77.21)
    assert len(calls) == 2


def test_nominatim_requests_are_spaced(monkeypatch):
    stamps = []

    def fake_get(url, params=None, headers=None, timeout=None):
        stamps.append(weather_client.time.monotonic())
        return _FakeResponse([{"lat": "1", "lon": "1", "display_name": params["q"], "address": {}}])

    monkeypatch.setattr(weather_client.requests, "get", fake_get)
    monkeypatch.setattr(weather_client, "NOMINATIM_MIN_INTERVAL_SECONDS", 0.2)
    weather_client.geocode_location_details("Pune")
    weather_client.geocode_location_details("Nagpur")
    assert stamps[1] - stamps[0] >= 0.19
