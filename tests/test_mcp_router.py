from weather_agent_core.mcp_registry import MCPCapabilityRegistry
from weather_agent_core.mcp_router import MCPCapabilityRouter


def registry():
    return MCPCapabilityRegistry([
        {"function": {"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object"}}},
        {"function": {"name": "get_forecast", "description": "Get forecast", "parameters": {"type": "object"}}},
        {"function": {"name": "search_weather", "description": "Search weather knowledge RAG evidence", "parameters": {"type": "object"}}},
    ])


def test_routes_requested_capabilities_to_discovered_tools():
    route = MCPCapabilityRouter(registry()).route(["forecast", "knowledge"])
    assert route.has_match
    assert route.selected == ("get_forecast", "search_weather")
    assert route.rejected == ()


def test_route_plan_ignores_optional_steps():
    route = MCPCapabilityRouter(registry()).route_plan({
        "steps": [
            {"required": True, "preferred_tools": ["forecast"]},
            {"required": False, "preferred_tools": ["knowledge"]},
        ]
    })
    assert route.selected == ("get_forecast",)


def test_unknown_capability_is_auditable():
    route = MCPCapabilityRouter(registry()).route(["database"])
    assert not route.has_match
    assert route.rejected == ("database",)
