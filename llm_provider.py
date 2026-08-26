"""LLM provider abstraction for local Ollama and Gemini API."""
from __future__ import annotations

import os
import time
from typing import Any


_GEMINI_CLIENT: Any | None = None


def provider_name() -> str:
    return os.environ.get("WEATHER_LLM_PROVIDER", "ollama").strip().lower()


def model_name() -> str:
    if provider_name() == "gemini":
        # Stable Flash is the default; callers can override this explicitly.
        return os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
    return os.environ.get("WEATHER_LLM_MODEL", "llama3.2:3b")


def _gemini_models() -> list[str]:
    """Return primary + fallback Gemini models without duplicates."""
    primary = model_name()
    configured = os.environ.get(
        "GEMINI_FALLBACK_MODELS",
        "gemini-3.5-flash-lite,gemini-2.5-flash-lite",
    )
    models = [primary]
    models.extend(item.strip() for item in configured.split(",") if item.strip())
    return list(dict.fromkeys(models))


def _gemini_client() -> Any:
    """Return one long-lived Gemini client per Python process."""
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in the process environment")
        _GEMINI_CLIENT = genai.Client(api_key=api_key)
    return _GEMINI_CLIENT


def _gemini_retryable(exc: Exception) -> bool:
    """Return True for transient Gemini capacity/server/rate-limit failures."""
    text = str(exc).upper()
    return any(
        marker in text
        for marker in (
            "503",
            "UNAVAILABLE",
            "429",
            "RESOURCE_EXHAUSTED",
            "500",
            "INTERNAL",
            "504",
            "DEADLINE_EXCEEDED",
        )
    )


def _generate_gemini(contents: str, *, temperature: float) -> tuple[str, str]:
    """Generate with retry + model fallback for transient provider failures."""
    from google.genai import types

    client = _gemini_client()
    errors: list[str] = []

    for model in _gemini_models():
        # Retry the same model briefly before falling back. This handles transient
        # 503/429 capacity spikes without hiding persistent model problems.
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        ),
                    ),
                )
                text = (response.text or "").strip()
                if not text:
                    raise RuntimeError(f"Gemini model {model} returned an empty response")
                return text, model
            except Exception as exc:
                errors.append(f"{model} attempt {attempt + 1}: {exc}")
                if not _gemini_retryable(exc):
                    # Authentication, invalid model, malformed request, etc. are
                    # deterministic failures and should not be retried blindly.
                    raise
                if attempt == 0:
                    time.sleep(1.5)

    raise RuntimeError(
        "All configured Gemini models failed after transient-error retries. "
        + " | ".join(errors)
    )


def generate_text(messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
    """Generate text through the configured provider.

    Provider selection is controlled by WEATHER_LLM_PROVIDER=ollama|gemini.
    Gemini uses GEMINI_API_KEY from the environment and automatically falls back
    across configured Flash models when a transient 503/429/5xx occurs.
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
            text, used_model = _generate_gemini(
                "\n\n".join(contents), temperature=temperature
            )
        except Exception as exc:
            raise RuntimeError(f"Gemini generation failed: {exc}") from exc

        # Expose the actual model used for observability without changing the
        # public return type expected by the rest of the application.
        os.environ["GEMINI_LAST_MODEL"] = used_model
        return text

    if provider != "ollama":
        raise ValueError(
            f"Unsupported WEATHER_LLM_PROVIDER: {provider!r}. Use 'ollama' or 'gemini'."
        )

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
