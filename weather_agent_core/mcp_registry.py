"""Dynamic MCP capability registry built from server-discovered schemas."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class MCPCapability:
    """Normalized description of one MCP tool capability."""

    name: str
    description: str
    schema: dict[str, Any]
    capabilities: frozenset[str]

    def supports(self, capability: str) -> bool:
        return capability.strip().lower() in self.capabilities


class MCPCapabilityRegistry:
    """Discover, validate and select MCP capabilities without hardcoding schemas."""

    def __init__(self, tools: Iterable[dict[str, Any]], allowed_tools: Iterable[str] | None = None) -> None:
        allowed = frozenset(allowed_tools) if allowed_tools is not None else None
        capabilities: list[MCPCapability] = []
        seen: set[str] = set()
        for tool in tools:
            function = tool.get("function") or {}
            name = str(function.get("name") or "").strip()
            if not name or name in seen or (allowed is not None and name not in allowed):
                continue
            seen.add(name)
            description = str(function.get("description") or "").strip()
            schema = function.get("parameters") or {"type": "object", "properties": {}}
            capabilities.append(
                MCPCapability(
                    name=name,
                    description=description,
                    schema=schema,
                    capabilities=self._infer_capabilities(name, description),
                )
            )
        self._capabilities = tuple(capabilities)
        self._by_name = {item.name: item for item in self._capabilities}

    @staticmethod
    def _infer_capabilities(name: str, description: str) -> frozenset[str]:
        text = f"{name} {description}".lower()
        groups: set[str] = {"weather"}
        rules = {
            "live": ("current", "live", "fresh", "today"),
            "forecast": ("forecast", "tomorrow", "7-day"),
            "alerts": ("alert", "hazard", "warning", "risk"),
            "knowledge": ("knowledge", "rag", "search", "retrieve", "evidence"),
            "location": ("location", "geocode", "city"),
            "database": ("database", "postgres", "lakebase", "health"),
            "sync": ("sync", "store", "persist"),
        }
        for group, keywords in rules.items():
            if any(keyword in text for keyword in keywords):
                groups.add(group)
        return frozenset(groups)

    @property
    def tools(self) -> tuple[MCPCapability, ...]:
        return self._capabilities

    @property
    def allowed_names(self) -> frozenset[str]:
        return frozenset(self._by_name)

    def get(self, name: str) -> MCPCapability | None:
        return self._by_name.get(name)

    def select(self, required: Iterable[str]) -> list[MCPCapability]:
        """Return capabilities matching any requested capability group, stably ordered."""
        wanted = {value.strip().lower() for value in required if value and value.strip()}
        if not wanted:
            return list(self._capabilities)
        return [item for item in self._capabilities if item.capabilities & wanted]

    def declarations(self) -> list[dict[str, Any]]:
        """Return Gemini-compatible function declarations for all permitted tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": item.name,
                    "description": item.description,
                    "parameters": item.schema,
                },
            }
            for item in self._capabilities
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "tool_count": len(self._capabilities),
            "tools": [
                {"name": item.name, "capabilities": sorted(item.capabilities)}
                for item in self._capabilities
            ],
        }


def build_registry(tools: list[dict[str, Any]], allowed_tools: Iterable[str] | None = None) -> MCPCapabilityRegistry:
    return MCPCapabilityRegistry(tools, allowed_tools=allowed_tools)
