"""Explicit state passed between agent layers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .evidence import Evidence, normalize_tool_result

@dataclass
class AgentState:
    query: str
    trace_id: str
    intent: str = "unknown"
    route: str = "agentic"
    plan: dict[str, Any] = field(default_factory=dict)
    required_tools: set[str] = field(default_factory=set)
    observations: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_observation(self, tool: str, arguments: dict[str, Any], result: Any) -> None:
        self.tool_calls.append({"name": tool, "arguments": arguments})
        self.observations.append({"tool": tool, "arguments": arguments, "result": result})
        if isinstance(result, dict) and result.get("success") is False:
            self.errors.append(f"{tool}: {result.get('error', 'tool failed')}")
        self.evidence.extend(normalize_tool_result(tool, result))
        if isinstance(result, dict):
            for source in result.get("sources") or []:
                if isinstance(source, dict) and source not in self.sources:
                    self.sources.append(source)

    @property
    def retrieval_failed(self) -> bool:
        return any(obs["tool"] in {"search_weather", "ask_weather"} and obs["tool"] in self.required_tools and isinstance(obs.get("result"), dict) and obs["result"].get("success") is False for obs in self.observations)

    @property
    def has_live_failure(self) -> bool:
        live_tools = {"get_weather", "get_forecast", "get_weather_alerts", "assess_weather_risk"}
        return any(obs["tool"] in live_tools and obs["tool"] in self.required_tools and isinstance(obs.get("result"), dict) and obs["result"].get("success") is False for obs in self.observations)

    def evidence_payload(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.evidence]
