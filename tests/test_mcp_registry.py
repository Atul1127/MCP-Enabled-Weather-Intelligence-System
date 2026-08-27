from weather_agent_core.mcp_registry import MCPCapabilityRegistry


def _tools():
    return [
        {"function": {"name": "get_weather", "description": "Get current weather and forecast.", "parameters": {"type": "object"}}},
        {"function": {"name": "search_weather", "description": "Search weather knowledge through RAG.", "parameters": {"type": "object"}}},
        {"function": {"name": "get_weather_alerts", "description": "Detect hazards and alerts.", "parameters": {"type": "object"}}},
        {"function": {"name": "get_weather", "description": "duplicate", "parameters": {}}},
    ]


def test_registry_filters_duplicates_and_policy():
    registry = MCPCapabilityRegistry(_tools(), allowed_tools={"get_weather", "search_weather"})
    assert registry.allowed_names == {"get_weather", "search_weather"}
    assert len(registry.tools) == 2


def test_registry_infers_capabilities_and_selects():
    registry = MCPCapabilityRegistry(_tools())
    assert registry.get("search_weather").supports("knowledge")
    assert registry.get("get_weather_alerts").supports("alerts")
    assert {item.name for item in registry.select(["knowledge"])} == {"search_weather"}


def test_registry_declarations_are_schema_preserving():
    registry = MCPCapabilityRegistry(_tools())
    declaration = next(item for item in registry.declarations() if item["function"]["name"] == "get_weather")
    assert declaration["function"]["parameters"] == {"type": "object"}
