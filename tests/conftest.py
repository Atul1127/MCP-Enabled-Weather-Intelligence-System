import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest


@pytest.fixture(autouse=True)
def _clear_weather_caches():
    """Keep the weather client's in-memory caches from leaking between tests."""
    import weather_client

    weather_client.clear_caches()
    yield
    weather_client.clear_caches()


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    """Each test starts with an empty rate-limit window."""
    from services import rate_limit

    rate_limit.reset()
    yield
    rate_limit.reset()
