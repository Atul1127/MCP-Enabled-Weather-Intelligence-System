from weather_agent_core.mcp_gateway import MCPGateway, MCPServer
from weather_agent_core.mcp_registry import MCPCapabilityRegistry


def registry(*names):
    return MCPCapabilityRegistry([
        {"function": {"name": name, "description": "weather capability", "parameters": {"type": "object"}}}
        for name in names
    ])


def test_gateway_routes_across_servers():
    gateway = MCPGateway([
        MCPServer("weather", object(), registry("get_weather")),
        MCPServer("knowledge", object(), registry("search_weather")),
    ])
    route = gateway.route(["get_weather", "knowledge"])
    assert "weather:get_weather" in route.selected
    assert "knowledge:search_weather" in route.selected
    assert not route.rejected


def test_gateway_resolves_qualified_tool():
    server = MCPServer("weather", object(), registry("get_weather"))
    gateway = MCPGateway([server])
    resolved_server, capability = gateway.find_tool("weather:get_weather")
    assert resolved_server is server
    assert capability is not None


def test_gateway_rejects_ambiguous_unqualified_tool():
    gateway = MCPGateway([
        MCPServer("a", object(), registry("search_weather")),
        MCPServer("b", object(), registry("search_weather")),
    ])
    server, capability = gateway.find_tool("search_weather")
    assert server is None
    assert capability is None
