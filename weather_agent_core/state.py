"""Explicit state passed between agent layers."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any
from .evidence import Evidence, normalize_tool_result

_CITATION_RE = re.compile(r"\[S(\d+)\]")
_SEARCH_TOOLS = {"search_weather", "ask_weather"}


def _renumber_citations(result: dict[str, Any], offset: int) -> None:
    """Shift [S1], [S2], ... in one search result by ``offset``, in place.

    Each search numbers its sources from S1, so two searches in one request
    would both produce an [S1]. Renumbering keeps every ID unique within the
    request before the model or the citation validator sees them.
    """
    if offset <= 0:
        return
    sources = [source for source in result.get("sources") or [] if isinstance(source, dict)]
    known = {str(source.get("citation")) for source in sources if source.get("citation")}

    def shift(match: re.Match[str]) -> str:
        label = f"S{match.group(1)}"
        return f"[S{int(match.group(1)) + offset}]" if label in known else match.group(0)

    if isinstance(result.get("context"), str):
        result["context"] = _CITATION_RE.sub(shift, result["context"])
    for source in sources:
        citation = str(source.get("citation") or "")
        if citation in known and citation[1:].isdigit():
            source["citation"] = f"S{int(citation[1:]) + offset}"

@dataclass
class AgentState:
    query: str
    trace_id: str
    intent: str = "unknown"
    route: str = "agentic"
    plan: dict[str, Any] = field(default_factory=dict)
    required_tool_groups: list[set[str]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_observation(self, tool: str, arguments: dict[str, Any], result: Any) -> None:
        if tool in _SEARCH_TOOLS and isinstance(result, dict):
            # Continue numbering after the sources already collected.
            _renumber_citations(result, len(self.sources))
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
    def required_requirements_satisfied(self) -> bool:
        for group in self.required_tool_groups:
            if not any(obs["tool"] in group and not (isinstance(obs.get("result"), dict) and obs["result"].get("success") is False) for obs in self.observations):
                return False
        return True

    @property
    def retrieval_failed(self) -> bool:
        return any(obs["tool"] in {"search_weather", "ask_weather"} and isinstance(obs.get("result"), dict) and obs["result"].get("success") is False for obs in self.observations)

    @property
    def has_live_failure(self) -> bool:
        live_tools = {"get_weather", "get_forecast", "get_weather_alerts", "assess_weather_risk"}
        return any(obs["tool"] in live_tools and isinstance(obs.get("result"), dict) and obs["result"].get("success") is False for obs in self.observations)

    def evidence_payload(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.evidence]
