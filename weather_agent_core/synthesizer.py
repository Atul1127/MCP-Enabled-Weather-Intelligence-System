"""Final Gemini synthesis over a unified evidence model."""
from __future__ import annotations
import asyncio
import json
from typing import Any
from google import genai
from google.genai import types

SYSTEM_PROMPT = """You are the final answer synthesizer for an Indian Weather Intelligence system.
Use only the supplied evidence. Never invent live weather values. Distinguish current
observations, forecasts, application-level risk assessments, official warnings, and
weather knowledge. Answer every part of the user's question. If evidence is missing
or a tool failed, say so clearly. Preserve supplied [S1], [S2] citation IDs for RAG
claims. Do not treat knowledge-base evidence as a live observation. Prefer the most
recent live evidence when answering current/forecast questions. Keep answers concise,
actionable, and explicit about uncertainty."""

class GeminiSynthesizer:
    def __init__(self, client: genai.Client, model: str):
        self.client, self.model = client, model

    async def synthesize(self, query: str, state: Any) -> str:
        payload = {
            "intent": state.intent,
            "route": state.route,
            "plan": state.plan,
            "evidence": state.evidence_payload(),
            "errors": state.errors,
        }
        prompt = f"User question:\n{query}\n\nUnified evidence:\n{json.dumps(payload, default=str)}"
        config_kwargs: dict[str, Any] = {
            "system_instruction": SYSTEM_PROMPT,
            "max_output_tokens": 900,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if not self.model.startswith(("gemini-3.5", "gemini-3.6", "gemini-3.7")):
            config_kwargs["temperature"] = 0
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini synthesizer returned an empty response")
        return text
