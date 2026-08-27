"""LangGraph state adapter built around the existing AgentState."""
from __future__ import annotations
from typing import Any, TypedDict

class GraphState(TypedDict, total=False):
    query: str
    trace_id: str
    intent: str
    route: str
    plan: dict[str, Any]
    observations: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    errors: list[str]
    metadata: dict[str, Any]
    messages: list[Any]
    next_action: str
    rounds: int
    answer: str
