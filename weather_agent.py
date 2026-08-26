"""Backward-compatible CLI wrapper around the Gemini MCP weather agent."""
from __future__ import annotations

import argparse
import asyncio

from agent import run_agent as _run_agent


async def run_agent(query: str) -> str:
    """Run the canonical Gemini MCP agent and return only its answer text."""
    result = await _run_agent(query)
    return str(result.get("answer") or "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Gemini Indian Weather MCP agent.")
    parser.add_argument("query", nargs="*", help="Natural-language weather question.")
    args = parser.parse_args()
    query = " ".join(args.query).strip() or input("Weather question: ").strip()
    print(asyncio.run(run_agent(query)))


if __name__ == "__main__":
    main()
