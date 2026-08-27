"""Backward-compatible entry point for the canonical Gemini RAG evaluation."""
from __future__ import annotations
import asyncio
from rag_llm_eval import main

if __name__ == "__main__":
    asyncio.run(main())
