"""Typed evidence layer shared by MCP, RAG, and Gemini synthesis."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
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


@dataclass(frozen=True)
class LiveWeatherEvidence(Evidence):
    kind: str = "live_weather"


@dataclass(frozen=True)
class RiskEvidence(Evidence):
    kind: str = "risk"


@dataclass(frozen=True)
class AlertEvidence(Evidence):
    kind: str = "alert"


@dataclass(frozen=True)
class RAGEvidence(Evidence):
    kind: str = "rag"


def normalize_tool_result(tool: str, result: Any) -> list[Evidence]:
    if not isinstance(result, dict) or result.get("success") is False:
        return []
    if tool in {"get_weather", "get_forecast"}:
        return [LiveWeatherEvidence(source="open-meteo", data=result, timestamp=result.get("observation_time"))]
    if tool == "assess_weather_risk":
        return [RiskEvidence(source="weather-risk-engine", data=result)]
    if tool == "get_weather_alerts":
        return [AlertEvidence(source="weather-risk-engine", data=result)]
    if tool in {"search_weather", "ask_weather"}:
        return [RAGEvidence(source="weather-knowledge-base", data={"query": result.get("query"), "context": result.get("context"), "documents": result.get("documents")}, citation=(result.get("sources") or [{}])[0].get("citation"))]
    return [Evidence(kind="tool", source=tool, data=result)]
