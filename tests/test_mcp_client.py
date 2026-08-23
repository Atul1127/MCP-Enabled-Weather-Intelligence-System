import asyncio

from mcp_client import call_tool, connect, discover_tools


def test_mcp_tool_discovery_and_weather_call():
    async def run():
        async with connect() as session:
            tools = await discover_tools(session)
            names = {tool["function"]["name"] for tool in tools}

            assert {"get_weather", "assess_weather_risk", "search_weather"}.issubset(names)

            result = await call_tool(
                session,
                "get_weather",
                {"location": "22.5726,88.3639"},
            )

            assert result["success"] is True
            assert result["location"]["latitude"] == 22.5726
            assert result["location"]["longitude"] == 88.3639
            assert "current" in result
            assert "daily" in result

            risk = await call_tool(
                session,
                "assess_weather_risk",
                {"location": "Kolkata", "activity": "cycling"},
            )

            assert risk["success"] is True
            assert risk["risk_level"] in {"LOW", "MODERATE", "HIGH"}
            assert "recommendation" in risk
            assert "factors" in risk

    asyncio.run(run())
