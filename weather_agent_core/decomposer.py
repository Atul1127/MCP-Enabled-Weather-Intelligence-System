"""Deterministic task decomposition for parallel-safe LangGraph execution."""
from __future__ import annotations

from typing import Any


def decompose(plan: dict[str, Any]) -> dict[str, Any]:
    """Annotate a plan with dependency-aware execution groups.

    Independent parallelizable steps share a group. Non-parallelizable steps
    become ordered groups. This keeps decomposition deterministic and lets the
    executor safely run independent MCP/RAG work concurrently.
    """
    steps = list(plan.get("steps", []))
    groups: list[list[str]] = []
    current: list[str] = []

    for step in steps:
        step_id = str(step.get("id", "step"))
        if step.get("parallelizable", True):
            current.append(step_id)
        else:
            if current:
                groups.append(current)
                current = []
            groups.append([step_id])
    if current:
        groups.append(current)

    return {
        **plan,
        "execution_groups": groups,
        "parallel_groups": [group for group in groups if len(group) > 1],
        "task_count": len(steps),
    }
