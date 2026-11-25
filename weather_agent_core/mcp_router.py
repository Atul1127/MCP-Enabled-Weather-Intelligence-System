"""Capability-aware routing for MCP tool selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .mcp_registry import MCPCapability, MCPCapabilityRegistry


@dataclass(frozen=True)
class MCPRoute:
    """A deterministic, auditable MCP routing decision."""

    requested: tuple[str, ...]
    selected: tuple[str, ...]
    rejected: tuple[str, ...]


class MCPCapabilityRouter:
    """Select permitted MCP capabilities from planner groups or tool names."""

    def __init__(self, registry: MCPCapabilityRegistry) -> None:
        self.registry = registry

    def route_plan(self, plan: dict[str, Any]) -> MCPRoute:
        """Route each required plan step to one available preferred tool.

        ``preferred_tools`` is an ordered OR-list, not a request to expose or
        execute every listed tool. Selecting one member of an alternative group
        prevents duplicate RAG/live calls while preserving deterministic fallback
        behavior when the preferred capability is unavailable.
        """
        requested: list[str] = []
        selected: list[str] = []
        rejected: list[str] = []

        for step in plan.get("steps", []):
            if not step.get("required", True):
                continue
            preferences = [str(value).strip().lower() for value in step.get("preferred_tools", []) if str(value).strip()]
            if not preferences:
                continue
            requested.extend(preferences)

            chosen: MCPCapability | None = None
            for value in preferences:
                exact = self.registry.get(value)
                if exact is not None:
                    chosen = exact
                    break

                matches = self.registry.select([value])
                if matches:
                    chosen = matches[0]
                    break

            if chosen is None:
                rejected.append("/".join(preferences))
                continue
            if chosen.name not in selected:
                selected.append(chosen.name)

        return MCPRoute(
            requested=tuple(dict.fromkeys(requested)),
            selected=tuple(selected),
            rejected=tuple(rejected),
        )
