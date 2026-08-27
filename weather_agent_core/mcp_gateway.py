"""Multi-server MCP gateway with capability-aware discovery and routing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .mcp_registry import MCPCapabilityRegistry
from .mcp_router import MCPRoute


@dataclass(frozen=True)
class MCPServer:
    """Logical MCP server backed by one initialized MCP session."""

    name: str
    session: Any
    registry: MCPCapabilityRegistry


class MCPGateway:
    """Route MCP capabilities across multiple independently discovered servers."""

    def __init__(self, servers: Iterable[MCPServer] = ()) -> None:
        self._servers = tuple(servers)
        names = [server.name for server in self._servers]
        if len(names) != len(set(names)):
            raise ValueError("MCP server names must be unique")

    @property
    def servers(self) -> tuple[MCPServer, ...]:
        return self._servers

    def route(self, required: Iterable[str]) -> MCPRoute:
        requested = tuple(dict.fromkeys(v.strip().lower() for v in required if v and v.strip()))
        selected: list[str] = []
        rejected: list[str] = []
        for value in requested:
            matches: list[str] = []
            for server in self._servers:
                exact = server.registry.get(value)
                if exact is not None:
                    matches.append(f"{server.name}:{exact.name}")
                else:
                    matches.extend(f"{server.name}:{item.name}" for item in server.registry.select([value]))
            if matches:
                selected.extend(item for item in matches if item not in selected)
            else:
                rejected.append(value)
        return MCPRoute(requested=requested, selected=tuple(selected), rejected=tuple(rejected))

    def find_tool(self, qualified_name: str):
        """Resolve ``server:tool`` or an unqualified unique tool name."""
        if ":" in qualified_name:
            server_name, tool_name = qualified_name.split(":", 1)
            for server in self._servers:
                if server.name == server_name:
                    return server, server.registry.get(tool_name)
            return None, None
        matches = [(server, server.registry.get(qualified_name)) for server in self._servers]
        matches = [(server, capability) for server, capability in matches if capability is not None]
        if len(matches) == 1:
            return matches[0]
        return None, None

    def summary(self) -> dict[str, Any]:
        return {
            "server_count": len(self._servers),
            "servers": [
                {"name": server.name, **server.registry.summary()}
                for server in self._servers
            ],
        }
