"""Capability-aware routing for MCP tool selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .mcp_registry import MCPCapability, MCPCapabilityRegistry


@dataclass(frozen=True)
class MCPRoute:
    """A deterministic, auditable MCP routing decision."""

    requested: tuple[str, ...]
    selected: tuple[str, ...]
    rejected: tuple[str, ...]

    @property
    def has_match(self) -> bool:
        return bool(self.selected)


class MCPCapabilityRouter:
    """Select permitted MCP capabilities from planner groups or tool names."""

    def __init__(self, registry: MCPCapabilityRegistry) -> None:
        self.registry = registry

    def route(self, required: Iterable[str]) -> MCPRoute:
        requested = tuple(dict.fromkeys(v.strip().lower() for v in required if v and v.strip()))
        selected_items: list[MCPCapability] = []
        for value in requested:
            exact = self.registry.get(value)
            if exact is not None:
                selected_items.append(exact)
                continue
            selected_items.extend(item for item in self.registry.select([value]) if item not in selected_items)
        selected = tuple(item.name for item in selected_items)
        rejected = tuple(value for value in requested if not any(item.name.lower() == value or item.supports(value) for item in selected_items))
        return MCPRoute(requested=requested, selected=selected, rejected=rejected)

    def route_plan(self, plan: dict[str, Any]) -> MCPRoute:
        """Route required preferred tools from a planner output."""
        requested: list[str] = []
        for step in plan.get("steps", []):
            if step.get("required", True):
                requested.extend(str(value) for value in step.get("preferred_tools", []))
        return self.route(requested)
