import asyncio

import weather_agent_core.executor as executor_module
from weather_agent_core.executor import MCPExecutor
from weather_agent_core.verifier import EvidenceVerifier


class _Call:
    def __init__(self, name, args):
        self.name = name
        self.args = args
        self.id = None


class _Session:
    pass


def test_executor_deduplicates_identical_calls_within_and_across_rounds(monkeypatch):
    calls = 0

    async def fake_call_tool(session, name, args):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return {"success": True, "value": 42}

    monkeypatch.setattr(executor_module, "call_tool", fake_call_tool)
    runner = MCPExecutor(_Session(), {"get_weather"}, max_retries=0)
    first = asyncio.run(
        runner.execute([
            _Call("get_weather", {"location": "Delhi"}),
            _Call("get_weather", {"location": "Delhi"}),
        ])
    )
    second = asyncio.run(runner.execute([_Call("get_weather", {"location": "Delhi"})]))
    assert calls == 1
    assert first[0][2] == first[1][2] == second[0][2]


def test_verifier_does_not_fail_for_optional_tool_error():
    plan = {
        "requires_live_data": True,
        "steps": [
            {"capability": "live_weather", "preferred_tools": ["get_weather"], "required": True},
            {"capability": "alerts", "preferred_tools": ["get_weather_alerts"], "required": False},
        ],
    }
    observations = [
        {"tool": "get_weather", "result": {"success": True}},
        {"tool": "get_weather_alerts", "result": {"success": False, "error": "failed"}},
    ]
    result = EvidenceVerifier().verify(plan, observations, [{"kind": "live_weather"}], ["get_weather_alerts: failed"])
    assert result["sufficient"] is True
    assert result["errors"]
