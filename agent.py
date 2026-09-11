"""CLI entry point and public compatibility API for the layered Gemini weather agent."""
from __future__ import annotations

import argparse
import asyncio
from typing import Any

from weather_agent_core import WeatherAgent


async def run_agent(query: str) -> dict[str, Any]:
    """Run the canonical WeatherAgent."""
    return await WeatherAgent().run(query)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Gemini MCP weather agent.")
    parser.add_argument("query", nargs="*", help="Natural-language weather question")
    args = parser.parse_args()
    query = " ".join(args.query).strip() or input("Weather question: ").strip()
    result = asyncio.run(run_agent(query))
    print(result["answer"])
    print(f"\ntrace_id={result['trace_id']} intent={result['intent']} route={result['route']}")


if __name__ == "__main__":
    main()
