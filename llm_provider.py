"""Gemini-only LLM provider for the weather intelligence system."""
from __future__ import annotations
import os
import time
from typing import Any

_GEMINI_CLIENT: Any | None = None

def provider_name() -> str:
    return "gemini"

def model_name() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

def _gemini_models() -> list[str]:
    primary = model_name()
    configured = os.environ.get("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite,gemini-2.5-flash-lite")
    return list(dict.fromkeys([primary] + [item.strip() for item in configured.split(",") if item.strip()]))

def _gemini_client() -> Any:
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in the process environment")
        _GEMINI_CLIENT = genai.Client(api_key=api_key)
    return _GEMINI_CLIENT

def _gemini_retryable(exc: Exception) -> bool:
    text = str(exc).upper()
    return any(marker in text for marker in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL", "504", "DEADLINE_EXCEEDED"))

def _gemini_thinking_level() -> str:
    level = os.environ.get("GEMINI_THINKING_LEVEL", "low").strip().lower()
    allowed = {"minimal", "low", "medium", "high"}
    if level not in allowed:
        raise ValueError(f"Unsupported GEMINI_THINKING_LEVEL: {level!r}. Use minimal, low, medium, or high.")
    return level

def _gemini_max_output_tokens() -> int:
    value = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "600"))
    if value < 64:
        raise ValueError("GEMINI_MAX_OUTPUT_TOKENS must be at least 64")
    return value

def _generate_gemini(contents: str, *, temperature: float) -> tuple[str, str]:
    """Generate with transient retry and Gemini model fallback."""
    from google.genai import types
    client = _gemini_client()
    errors: list[str] = []
    for model in _gemini_models():
        config_kwargs: dict[str, Any] = {
            "max_output_tokens": _gemini_max_output_tokens(),
            "thinking_config": types.ThinkingConfig(thinking_level=_gemini_thinking_level()),
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        # Gemini 3.5+ removed the legacy temperature/top-p/top-k sampling controls.
        if not model.startswith(("gemini-3.5", "gemini-3.6", "gemini-3.7")):
            config_kwargs["temperature"] = temperature
        config = types.GenerateContentConfig(**config_kwargs)
        for attempt in range(2):
            try:
                response = client.models.generate_content(model=model, contents=contents, config=config)
                text = (response.text or "").strip()
                if not text:
                    raise RuntimeError(f"Gemini model {model} returned an empty response")
                return text, model
            except Exception as exc:
                errors.append(f"{model} attempt {attempt + 1}: {exc}")
                if not _gemini_retryable(exc):
                    raise
                if attempt == 0:
                    time.sleep(1.0)
    raise RuntimeError("All configured Gemini models failed after retries. " + " | ".join(errors))

def generate_text(messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
    """Generate grounded text through Gemini only."""
    contents: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        prefix = "System instructions:\n" if role == "system" else "User:\n" if role == "user" else f"{role.title()}:\n"
        contents.append(prefix + content)
    try:
        text, used_model = _generate_gemini("\n\n".join(contents), temperature=temperature)
    except Exception as exc:
        raise RuntimeError(f"Gemini generation failed: {exc}") from exc
    os.environ["GEMINI_LAST_MODEL"] = used_model
    return text
