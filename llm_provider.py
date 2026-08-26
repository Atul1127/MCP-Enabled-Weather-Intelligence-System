"""LLM provider abstraction for local Ollama and Gemini API."""
from __future__ import annotations

import os
from typing import Any


PROVIDER = os.environ.get("WEATHER_LLM_PROVIDER", "ollama").strip().lower()
OLLAMA_MODEL = os.environ.get("WEATHER_LLM_MODEL", "llama3.2:3b")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def _gemini_client():
    from google import genai
    return genai.Client()


def generate_text(messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
    """Generate text through the configured provider.

    Provider selection is controlled by WEATHER_LLM_PROVIDER=ollama|gemini.
    Gemini reads GEMINI_API_KEY from the environment automatically.
    """
    if PROVIDER == "gemini":
        contents = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            prefix = "System instructions:\n" if role == "system" else ("User:\n" if role == "user" else f"{role.title()}:\n")
            contents.append(prefix + content)
        response = _gemini_client().models.generate_content(
            model=GEMINI_MODEL,
            contents="\n\n".join(contents),
            config={"temperature": temperature},
        )
        return (response.text or "").strip()

    import ollama
    response = ollama.chat(model=OLLAMA_MODEL, messages=messages, options={"temperature": temperature})
    return str(response["message"]["content"]).strip()


def provider_name() -> str:
    return PROVIDER


def model_name() -> str:
    return GEMINI_MODEL if PROVIDER == "gemini" else OLLAMA_MODEL
