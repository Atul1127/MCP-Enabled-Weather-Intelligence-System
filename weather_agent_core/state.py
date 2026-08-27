"""Explicit state passed between agent layers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    query: str
    trace_id: str
    intent: str = "unknown"
    route: str = "agentic"
    plan: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_observation(self, tool: str, arguments: dict[str, Any], result: Any) -> None:
        self.tool_calls.append({"name": tool, "arguments": arguments})
        self.observations.append({"tool": tool, "arguments": arguments, "result": result})

    @property
    def retrieval_failed(self) -> bool:
        return any(
            obs["tool"] in {"search_weather", "ask_weather"}
            and isinstance(obs.get("result"), dict)
            and not obs["result"].get("success", False)
            for obs in self.observations
        )
