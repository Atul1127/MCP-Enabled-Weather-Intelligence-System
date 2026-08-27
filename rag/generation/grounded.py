"""Grounded Gemini generation for RAG evidence."""
from __future__ import annotations
import asyncio
from typing import Any
from google import genai
from google.genai import types

SYSTEM_PROMPT = """You are a grounded weather knowledge assistant. Use only the supplied retrieved evidence. Do not invent facts. Cite factual claims with the supplied [S1], [S2] identifiers. If the evidence is insufficient, say so. Distinguish knowledge-base guidance from live forecasts and official warnings."""

async def generate(client: genai.Client, model: str, query: str, context: str) -> str:
    response = await asyncio.to_thread(
        client.models.generate_content,
        model=model,
        contents=f"Question:\n{query}\n\nRetrieved evidence:\n{context}",
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0, max_output_tokens=800),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty RAG answer")
    return text
