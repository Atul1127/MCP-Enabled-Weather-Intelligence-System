"""Backward-compatible entry point for the canonical Gemini agent benchmark."""
from __future__ import annotations
import asyncio
from agent_benchmark import main

if __name__ == "__main__":
    asyncio.run(main())
