import asyncio
from types import SimpleNamespace

import pytest

from weather_agent_core.synthesizer import GeminiSynthesizer


class FakeModels:
    def __init__(self, text):
        self.text = text

    def generate_content(self, **kwargs):
        return SimpleNamespace(text=self.text)


class FakeClient:
    def __init__(self, text):
        self.models = FakeModels(text)


def test_structured_response_accepts_valid_contract():
    result = asyncio.run(
        GeminiSynthesizer(
            FakeClient('{"answer":"Rain is likely.","confidence":0.9,"citations":["S1"],"warnings":[]}'),
            "test-model",
        ).synthesize_structured("weather?", SimpleNamespace(intent="x", route="mcp", plan={}, evidence_payload=lambda: [], errors=[]))
    )
    assert result == {
        "answer": "Rain is likely.",
        "confidence": 0.9,
        "citations": ["S1"],
        "warnings": [],
    }


def test_structured_response_rejects_invalid_confidence():
    client = FakeClient('{"answer":"ok","confidence":2,"citations":[],"warnings":[]}')
    state = SimpleNamespace(intent="x", route="mcp", plan={}, evidence_payload=lambda: [], errors=[])
    with pytest.raises(RuntimeError, match="confidence"):
        asyncio.run(GeminiSynthesizer(client, "test-model").synthesize_structured("weather?", state))


def test_structured_response_rejects_invalid_json():
    client = FakeClient("not-json")
    state = SimpleNamespace(intent="x", route="mcp", plan={}, evidence_payload=lambda: [], errors=[])
    with pytest.raises(RuntimeError, match="invalid structured JSON"):
        asyncio.run(GeminiSynthesizer(client, "test-model").synthesize_structured("weather?", state))
