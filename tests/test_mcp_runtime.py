"""The MCP server must start once per worker and be reused across requests."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

import app
from mcp_integration.runtime import MCPRuntime


@pytest.fixture
def runtime():
    rt = MCPRuntime()
    yield rt
    rt.close()


async def _tool_names(session):
    result = await session.list_tools()
    return sorted(tool.name for tool in result.tools)


def test_session_is_started_once_and_reused(runtime):
    first = runtime.run(_tool_names, timeout=60)
    session = runtime._session
    second = runtime.run(_tool_names, timeout=60)
    assert "get_weather" in first and first == second
    assert runtime._session is session  # no second server process


def test_concurrent_requests_share_one_server(runtime):
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: runtime.run(_tool_names, timeout=60), range(4)))
    assert all(r == results[0] for r in results)
    holders = [runtime._holder]
    assert len(holders) == 1 and not runtime._holder.done()


def test_dead_server_is_restarted(runtime):
    runtime.run(_tool_names, timeout=60)
    old = runtime._session
    runtime._reset()  # simulate the server going away
    assert "get_weather" in runtime.run(_tool_names, timeout=60)
    assert runtime._session is not old


def test_agent_timeout_returns_json_504(monkeypatch):
    class SlowRuntime:
        def run(self, work, *, timeout):
            raise TimeoutError("too slow")

    monkeypatch.setattr(app, "mcp_runtime", SlowRuntime())
    response = app.app.test_client().post("/weather/agent", json={"query": "Weather in Kolkata"})
    assert response.status_code == 504
    assert "too long" in response.get_json()["error"]
