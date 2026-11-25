"""The agent reuses one MCP server process across requests."""
import asyncio
import threading
import time

from mcp_integration import session_pool
from mcp_integration.mcp_client import discover_tools, read_resource


async def _one_request():
    async with session_pool.agent_session() as session:
        tools = await discover_tools(session)
        capabilities = await read_resource(session, "weather://capabilities")
        return {tool["function"]["name"] for tool in tools}, capabilities


def test_server_is_reused_across_requests(monkeypatch):
    monkeypatch.setenv("WEATHER_MCP_PERSISTENT", "1")
    session_pool.shutdown()
    try:
        started = time.perf_counter()
        names, capabilities = asyncio.run(_one_request())
        first_session = session_pool._POOL._session
        cold = time.perf_counter() - started

        started = time.perf_counter()
        names_again, _ = asyncio.run(_one_request())
        warm = time.perf_counter() - started

        assert "get_weather" in names and names == names_again
        assert capabilities and "get_weather" in capabilities[0]["text"]
        assert session_pool._POOL._session is first_session
        assert warm < cold
    finally:
        session_pool.shutdown()


def test_concurrent_requests_share_the_server(monkeypatch):
    monkeypatch.setenv("WEATHER_MCP_PERSISTENT", "1")
    session_pool.shutdown()
    results, errors = [], []

    def worker():
        try:
            results.append(asyncio.run(_one_request())[0])
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    try:
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        assert not errors
        assert len(results) == 4 and all("get_weather" in names for names in results)
    finally:
        session_pool.shutdown()


def test_server_restarts_after_shutdown(monkeypatch):
    monkeypatch.setenv("WEATHER_MCP_PERSISTENT", "1")
    session_pool.shutdown()
    try:
        asyncio.run(_one_request())
        first = session_pool._POOL._session
        session_pool.shutdown()
        names, _ = asyncio.run(_one_request())
        assert "get_weather" in names
        assert session_pool._POOL._session is not first
    finally:
        session_pool.shutdown()


def test_per_request_mode_still_works(monkeypatch):
    monkeypatch.setenv("WEATHER_MCP_PERSISTENT", "0")
    names, _ = asyncio.run(_one_request())
    assert "get_weather" in names
