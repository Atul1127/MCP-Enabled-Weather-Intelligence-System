"""Typed evidence layer shared by MCP, RAG, and Gemini synthesis."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any

@dataclass(frozen=True)
class Evidence:
    kind: str
    source: str
    data: dict[str, Any]
    confidence: float | None = None
    timestamp: str | None = None
    citation: str | None = None
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

class LiveWeatherEvidence(Evidence):
    def __init__(self, source: str, data: dict[str, Any], confidence: float | None = None, timestamp: str | None = None, citation: str | None = None):
        super().__init__("live_weather", source, data, confidence, timestamp, citation)

class RiskEvidence(Evidence):
    def __init__(self, source: str, data: dict[str, Any], confidence: float | None = None, timestamp: str | None = None, citation: str | None = None):
        super().__init__("risk", source, data, confidence, timestamp, citation)

class AlertEvidence(Evidence):
    def __init__(self, source: str, data: dict[str, Any], confidence: float | None = None, timestamp: str | None = None, citation: str | None = None):
        super().__init__("alert", source, data, confidence, timestamp, citation)

class RAGEvidence(Evidence):
    def __init__(self, source: str, data: dict[str, Any], confidence: float | None = None, timestamp: str | None = None, citation: str | None = None):
        super().__init__("rag", source, data, confidence, timestamp, citation)

def normalize_tool_result(tool: str, result: Any) -> list[Evidence]:
    if not isinstance(result, dict) or result.get("success") is False:
        return []
    if tool in {"get_weather", "get_forecast"}:
        return [LiveWeatherEvidence("open-meteo", result, timestamp=result.get("observation_time"))]
    if tool == "assess_weather_risk":
        return [RiskEvidence("weather-risk-engine", result)]
    if tool == "get_weather_alerts":
        return [AlertEvidence("weather-risk-engine", result)]
    if tool in {"search_weather", "ask_weather"}:
        return [RAGEvidence("weather-knowledge-base", {"query": result.get("query"), "context": result.get("context"), "documents": result.get("documents"), "sources": result.get("sources")})]
    return [Evidence("tool", tool, result)]
