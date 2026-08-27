"""CLI entry point for the layered Gemini weather agent.

The implementation lives in weather_agent_core so orchestration concerns are
separated from the executable entry point.
"""
from __future__ import annotations

import argparse
import asyncio

from weather_agent_core import WeatherAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Gemini MCP weather agent.")
    parser.add_argument("query", nargs="*", help="Natural-language weather question")
    args = parser.parse_args()
    query = " ".join(args.query).strip() or input("Weather question: ").strip()
    result = asyncio.run(WeatherAgent().run(query))
    print(result["answer"])
    print(f"\ntrace_id={result['trace_id']} intent={result['intent']} route={result['route']}")


if __name__ == "__main__":
    main()
