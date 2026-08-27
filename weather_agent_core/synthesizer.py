"""Final answer synthesis and grounding policy."""
from __future__ import annotations

import asyncio
from typing import Any

from google import genai
from google.genai import types


SYSTEM_PROMPT = """You are the final answer synthesizer for an Indian Weather Intelligence system.
Use only the supplied MCP observations as evidence. Never invent live weather values.
Distinguish current observations, forecasts, application-level risk assessments, and
knowledge-base evidence. Answer every part of the user's question. If evidence is
missing or a tool failed, state that clearly. When knowledge-base sources are supplied,
preserve their [S1], [S2] citation IDs and cite factual claims. Keep the answer concise
and actionable."""


class GeminiSynthesizer:
    def __init__(self, client: genai.Client, model: str):
        self.client = client
        self.model = model

    async def synthesize(self, query: str, state: Any) -> str:
        evidence = {
            "intent": state.intent,
            "plan": state.plan,
            "observations": state.observations,
            "sources": state.sources,
        }
        prompt = (
            f"User question:\n{query}\n\n"
            f"Execution evidence (JSON):\n{__import__('json').dumps(evidence, default=str)}"
        )
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0,
                max_output_tokens=800,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini synthesizer returned an empty response")
        return text
