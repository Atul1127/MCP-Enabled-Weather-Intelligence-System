"""Central Gemini provider for text, tool, and structured generation."""
from __future__ import annotations

import os
import time
from typing import Any

_GEMINI_CLIENT: Any | None = None
_GEMINI_RETRY_ATTEMPTS = 2
_RETRYABLE_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL", "504", "DEADLINE_EXCEEDED")
_GEMINI_3_PREFIXES = ("gemini-3.5", "gemini-3.6", "gemini-3.7")


def provider_name() -> str:
    return "gemini"


def model_name() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")


def _gemini_models(primary: str | None = None) -> list[str]:
    primary = primary or model_name()
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
    return any(marker in text for marker in _RETRYABLE_MARKERS)


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


def _base_config_kwargs() -> dict[str, Any]:
    from google.genai import types
    return {
        "max_output_tokens": _gemini_max_output_tokens(),
        "thinking_config": types.ThinkingConfig(thinking_level=_gemini_thinking_level()),
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
    }


def _generate(contents: Any, *, config_kwargs: dict[str, Any], require_text: bool = False, client: Any | None = None, models: list[str] | None = None) -> tuple[Any, str]:
    """Generate through one retry/fallback policy for every Gemini call type."""
    from google.genai import types
    client = client or _gemini_client()
    errors: list[str] = []
    for model in models or _gemini_models():
        kwargs = dict(config_kwargs)
        if model.startswith(_GEMINI_3_PREFIXES):
            kwargs.pop("temperature", None)
        else:
            kwargs.setdefault("temperature", 0.0)
        config = types.GenerateContentConfig(**kwargs)
        for attempt in range(_GEMINI_RETRY_ATTEMPTS):
            try:
                response = client.models.generate_content(model=model, contents=contents, config=config)
                if require_text and not (response.text or "").strip():
                    raise RuntimeError(f"Gemini model {model} returned an empty response")
                os.environ["GEMINI_LAST_MODEL"] = model
                return response, model
            except Exception as exc:
                errors.append(f"{model} attempt {attempt + 1}: {exc}")
                if not _gemini_retryable(exc):
                    raise
                if attempt + 1 < _GEMINI_RETRY_ATTEMPTS:
                    time.sleep(1.0)
    raise RuntimeError("All configured Gemini models failed after retries. " + " | ".join(errors))


def generate_text(messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
    """Generate plain text through the shared Gemini provider."""
    contents: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        prefix = "System instructions:\n" if role == "system" else "User:\n" if role == "user" else f"{role.title()}:\n"
        contents.append(prefix + content)
    config_kwargs = _base_config_kwargs()
    config_kwargs["temperature"] = temperature
    response, _ = _generate("\n\n".join(contents), config_kwargs=config_kwargs, require_text=True)
    return (response.text or "").strip()


def generate_structured(*, contents: Any, system_instruction: str, response_schema: dict[str, Any], max_output_tokens: int = 900, client: Any | None = None, model: str | None = None) -> Any:
    """Generate structured JSON using the shared retry/fallback policy."""
    config_kwargs = _base_config_kwargs()
    config_kwargs.update({"system_instruction": system_instruction, "max_output_tokens": max_output_tokens, "response_mime_type": "application/json", "response_schema": response_schema})
    response, _ = _generate(contents, config_kwargs=config_kwargs, require_text=True, client=client, models=_gemini_models(model))
    return response


def generate_with_tools(*, contents: Any, system_instruction: str, declarations: list[Any], max_output_tokens: int = 700, client: Any | None = None, model: str | None = None) -> Any:
    """Generate a tool-selection response using the shared Gemini policy."""
    from google.genai import types
    config_kwargs = _base_config_kwargs()
    config_kwargs.update({"system_instruction": system_instruction, "max_output_tokens": max_output_tokens, "tools": [types.Tool(function_declarations=declarations)]})
    response, _ = _generate(contents, config_kwargs=config_kwargs, client=client, models=_gemini_models(model))
    return response


class GeminiProvider:
    """Dependency-injection wrapper around the shared Gemini retry/fallback policy."""
    def __init__(self, model: str | None = None, client: Any | None = None):
        self.model = model or model_name()
        self._client = client

    @property
    def client(self) -> Any:
        """Return the shared client, creating it only when first needed."""
        if self._client is None:
            self._client = _gemini_client()
        return self._client

    def generate_text(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        contents = []
        for message in messages:
            role = message.get("role", "user")
            prefix = "System instructions:\n" if role == "system" else "User:\n" if role == "user" else f"{role.title()}:\n"
            contents.append(prefix + message.get("content", ""))
        config_kwargs = _base_config_kwargs()
        config_kwargs["temperature"] = temperature
        response, _ = _generate("\n\n".join(contents), config_kwargs=config_kwargs, require_text=True, client=self.client, models=_gemini_models(self.model))
        return (response.text or "").strip()

    def generate_structured(self, *, contents: Any, system_instruction: str, response_schema: dict[str, Any], max_output_tokens: int = 900) -> Any:
        return generate_structured(contents=contents, system_instruction=system_instruction, response_schema=response_schema, max_output_tokens=max_output_tokens, client=self.client, model=self.model)

    def generate_with_tools(self, *, contents: Any, system_instruction: str, declarations: list[Any], max_output_tokens: int = 700) -> Any:
        return generate_with_tools(contents=contents, system_instruction=system_instruction, declarations=declarations, max_output_tokens=max_output_tokens, client=self.client, model=self.model)
