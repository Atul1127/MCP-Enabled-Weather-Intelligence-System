"""CLI entry point and compatibility API for the layered Gemini weather agent."""
from __future__ import annotations
import argparse
import asyncio
import re
from typing import Any
from weather_agent_core import WeatherAgent


def _is_comparison_query(query: str) -> bool:
    """Compatibility predicate used by benchmarks and the CLI contract."""
    text = " ".join((query or "").lower().split())
    if any(marker in text for marker in ("compare ", "comparison", "versus", " vs ", "which is better", "better for")):
        return True
    return bool(re.search(r"\b(?:between|or)\b.+\b(?:and|or)\b", text))


async def run_agent(query: str) -> dict[str, Any]:
    """Run the canonical WeatherAgent; retained for benchmark/integration compatibility."""
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
