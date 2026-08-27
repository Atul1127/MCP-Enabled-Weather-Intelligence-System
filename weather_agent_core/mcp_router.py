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
    """Select permitted MCP capabilities from planner capability groups.

    Routing is deliberately deterministic: the planner supplies semantic
    capability groups and the registry supplies discovered MCP schemas.
    The router never invents a tool name or bypasses the MCP client.
    """

    def __init__(self, registry: MCPCapabilityRegistry) -> None:
        self.registry = registry

    def route(self, required: Iterable[str]) -> MCPRoute:
        requested = tuple(dict.fromkeys(v.strip().lower() for v in required if v and v.strip()))
        matches: list[MCPCapability] = self.registry.select(requested)
        selected = tuple(item.name for item in matches)
        rejected = tuple(value for value in requested if not any(item.supports(value) for item in matches))
        return MCPRoute(requested=requested, selected=selected, rejected=rejected)

    def route_plan(self, plan: dict[str, Any]) -> MCPRoute:
        """Route all preferred tool groups in a planner output."""
        groups: list[str] = []
        for step in plan.get("steps", []):
            if not step.get("required", True):
                continue
            groups.extend(str(value) for value in step.get("preferred_tools", []))
        return self.route(groups)
