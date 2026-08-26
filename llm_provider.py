"""LLM provider abstraction for local Ollama and Gemini API."""
from __future__ import annotations

import os
from typing import Any


_GEMINI_CLIENT: Any | None = None


def provider_name() -> str:
    return os.environ.get("WEATHER_LLM_PROVIDER", "ollama").strip().lower()


def model_name() -> str:
    if provider_name() == "gemini":
        return os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")
    return os.environ.get("WEATHER_LLM_MODEL", "llama3.2:3b")


def _gemini_client() -> Any:
    """Return one long-lived Gemini client per Python process.

    Do not create a temporary client inline with ``Client().models...`` because
    the SDK client owns an HTTPX connection pool and can be closed before the
    request finishes when the temporary object is garbage-collected.
    """
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in the process environment")
        _GEMINI_CLIENT = genai.Client(api_key=api_key)
    return _GEMINI_CLIENT


def generate_text(messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
    """Generate text through the configured provider.

    Provider selection is controlled by WEATHER_LLM_PROVIDER=ollama|gemini.
    Gemini uses GEMINI_API_KEY from the environment.
    """
    provider = provider_name()

    if provider == "gemini":
        contents: list[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            prefix = (
                "System instructions:\n"
                if role == "system"
                else "User:\n"
                if role == "user"
                else f"{role.title()}:\n"
            )
            contents.append(prefix + content)

        try:
            response = _gemini_client().models.generate_content(
                model=model_name(),
                contents="\n\n".join(contents),
                config={"temperature": temperature},
            )
        except Exception as exc:
            raise RuntimeError(f"Gemini generation failed: {exc}") from exc

        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned an empty response")
        return text

    if provider != "ollama":
        raise ValueError(f"Unsupported WEATHER_LLM_PROVIDER: {provider!r}. Use 'ollama' or 'gemini'.")

    import ollama

    try:
        response = ollama.chat(
            model=model_name(),
            messages=messages,
            options={"temperature": temperature},
        )
    except Exception as exc:
        raise RuntimeError(f"Ollama generation failed: {exc}") from exc

    return str(response["message"]["content"]).strip()
