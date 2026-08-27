"""Single Gemini gateway for text, tool-calling, and structured generation."""
from __future__ import annotations

import os
import time
from typing import Any, Sequence

_GEMINI_CLIENT: Any | None = None


def provider_name() -> str:
    return "gemini"


def model_name() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")


def _gemini_models() -> list[str]:
    primary = model_name()
    configured = os.environ.get(
        "GEMINI_FALLBACK_MODELS",
        "gemini-3.5-flash-lite,gemini-2.5-flash-lite",
    )
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
    return any(
        marker in text
        for marker in (
            "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500",
            "INTERNAL", "504", "DEADLINE_EXCEEDED",
        )
    )


def _gemini_thinking_level() -> str:
    level = os.environ.get("GEMINI_THINKING_LEVEL", "low").strip().lower()
    allowed = {"minimal", "low", "medium", "high"}
    if level not in allowed:
        raise ValueError(
            f"Unsupported GEMINI_THINKING_LEVEL: {level!r}. "
            "Use minimal, low, medium, or high."
        )
    return level


def _gemini_max_output_tokens() -> int:
    value = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "700"))
    if value < 64:
        raise ValueError("GEMINI_MAX_OUTPUT_TOKENS must be at least 64")
    return value


def _base_config(*, temperature: float | None = 0.0, **extra: Any) -> Any:
    from google.genai import types

    kwargs: dict[str, Any] = {
        "max_output_tokens": _gemini_max_output_tokens(),
        "thinking_config": types.ThinkingConfig(
            thinking_level=_gemini_thinking_level()
        ),
        **extra,
    }
    # Gemini 3.5+ removed legacy sampling controls.
    if temperature is not None and not model_name().startswith(
        ("gemini-3.5", "gemini-3.6", "gemini-3.7")
    ):
        kwargs["temperature"] = temperature
    return types.GenerateContentConfig(**kwargs)


def _generate_with_fallback(
    *,
    contents: Any,
    config_factory: Any,
) -> tuple[Any, str]:
    client = _gemini_client()
    errors: list[str] = []

    for model in _gemini_models():
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config_factory(model),
                )
                os.environ["GEMINI_LAST_MODEL"] = model
                return response, model
            except Exception as exc:
                errors.append(f"{model} attempt {attempt + 1}: {exc}")
                if not _gemini_retryable(exc):
                    raise
                if attempt == 0:
                    time.sleep(1.0)

    raise RuntimeError(
        "All configured Gemini models failed after retries. " + " | ".join(errors)
    )


def _config_for_model(
    model: str,
    *,
    temperature: float | None = 0.0,
    **extra: Any,
) -> Any:
    from google.genai import types

    kwargs: dict[str, Any] = {
        "max_output_tokens": _gemini_max_output_tokens(),
        "thinking_config": types.ThinkingConfig(
            thinking_level=_gemini_thinking_level()
        ),
        **extra,
    }
    if temperature is not None and not model.startswith(
        ("gemini-3.5", "gemini-3.6", "gemini-3.7")
    ):
        kwargs["temperature"] = temperature
    return types.GenerateContentConfig(**kwargs)


def generate_text(messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
    """Generate plain text through the single Gemini provider."""
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

    def factory(model: str) -> Any:
        return _config_for_model(model, temperature=temperature)

    try:
        response, _ = _generate_with_fallback(
            contents="\n\n".join(contents),
            config_factory=factory,
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini generation failed: {exc}") from exc

    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response")
    return text


def generate_with_tools(
    contents: Sequence[Any],
    *,
    declarations: Sequence[Any],
    system_instruction: str,
    temperature: float | None = 0.0,
) -> Any:
    """Generate with Gemini function declarations; callers execute returned calls."""
    from google.genai import types

    def factory(model: str) -> Any:
        return _config_for_model(
            model,
            temperature=temperature,
            system_instruction=system_instruction,
            tools=[types.Tool(function_declarations=list(declarations))],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

    response, _ = _generate_with_fallback(
        contents=list(contents),
        config_factory=factory,
    )
    return response


def generate_structured(
    contents: Any,
    *,
    system_instruction: str,
    response_schema: dict[str, Any],
    temperature: float | None = 0.0,
) -> str:
    """Generate JSON text using Gemini's response schema."""
    def factory(model: str) -> Any:
        return _config_for_model(
            model,
            temperature=temperature,
            system_instruction=system_instruction,
            automatic_function_calling=__import__(
                "google.genai", fromlist=["types"]
            ).types.AutomaticFunctionCallingConfig(disable=True),
            response_mime_type="application/json",
            response_schema=response_schema,
        )

    response, _ = _generate_with_fallback(
        contents=contents,
        config_factory=factory,
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini structured generation returned an empty response")
    return text
