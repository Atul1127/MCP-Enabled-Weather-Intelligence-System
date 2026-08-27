"""Declarative MCP server configuration for the gateway."""
from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    command: str
    args: tuple[str, ...]
    enabled: bool = True


def configured_servers() -> tuple[MCPServerConfig, ...]:
    """Return enabled server definitions without opening connections."""
    python = os.environ.get("WEATHER_PYTHON") or os.environ.get("PYTHON", "python")
    return (
        MCPServerConfig("weather", python, ("mcp_server.py",)),
    )
