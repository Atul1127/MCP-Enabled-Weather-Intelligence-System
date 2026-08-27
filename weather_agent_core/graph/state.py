"""Runtime state passed between LangGraph nodes."""
from __future__ import annotations
from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    query: str
    trace_id: str
    intent: str
    route: str
    plan: dict[str, Any]
    agent_state: Any
    contents: list[Any]
    declarations: list[Any]
    calls: list[Any]
    pending_calls: list[Any]
    candidate: Any
    next_action: str
    rounds: int
    answer: str
    observations: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    errors: list[str]
    verification: dict[str, Any]
    retry_reason: str | None
    retry_count: int
