"""Single Gemini gateway for text, tool-calling, and structured generation."""
from __future__ import annotations

import os
import re
import time
from typing import Any, Sequence

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass

_GEMINI_CLIENT: Any | None = None


def provider_name() -> str:
    return "gemini"


def model_name() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")


def _gemini_models(primary: str | None = None) -> list[str]:
    primary_name = primary or model_name()
    configured = os.environ.get("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite")
    return list(dict.fromkeys([primary_name] + [item.strip() for item in configured.split(",") if item.strip()]))


def _gemini_client() -> Any:
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in the process environment")
        _GEMINI_CLIENT = genai.Client(api_key=api_key)
    return _GEMINI_CLIENT


_TRANSIENT_STATUS_CODES = {500, 502, 503, 504}
_TRANSIENT_STATUS_NAMES = {"UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED"}


def _gemini_error_kind(exc: Exception) -> str:
    """Classify a Gemini error from its HTTP status code and status name.

    google-genai raises errors.APIError with ``code`` (HTTP status) and
    ``status`` (e.g. "RESOURCE_EXHAUSTED"). Bare numbers in the message are
    not used, so a message that merely mentions "500 tokens" is not mistaken
    for a server error.
    """
    raw_code = getattr(exc, "status_code", None)
    if raw_code is None:
        raw_code = getattr(exc, "code", None)
    try:
        code = int(raw_code) if raw_code is not None else None
    except (TypeError, ValueError):
        code = None
    status_name = str(getattr(exc, "status", "") or "").upper()
    message = str(exc).upper()

    if status_name == "RESOURCE_EXHAUSTED" or re.search(r"\bRESOURCE_EXHAUSTED\b|\bQUOTA[ _]EXCEEDED\b|\bQUOTA EXHAUSTED\b", message):
        return "quota"
    if code == 429 or re.search(r"\bRATE[ _]LIMIT|\bTOO MANY REQUESTS\b", message):
        return "rate_limit"
    if code in _TRANSIENT_STATUS_CODES or status_name in _TRANSIENT_STATUS_NAMES:
        return "transient"
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "transient"
    if code is None and re.search(r"\b(?:UNAVAILABLE|DEADLINE_EXCEEDED)\b", message):
        return "transient"
    return "permanent"


def _gemini_retryable(exc: Exception) -> bool:
    return _gemini_error_kind(exc) in {"transient", "rate_limit"}


def _gemini_thinking_level() -> str:
    level = os.environ.get("GEMINI_THINKING_LEVEL", "low").strip().lower()
    allowed = {"minimal", "low", "medium", "high"}
    if level not in allowed:
        raise ValueError(f"Unsupported GEMINI_THINKING_LEVEL: {level!r}. Use minimal, low, medium, or high.")
    return level


def _gemini_max_output_tokens() -> int:
    value = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "700"))
    if value < 64:
        raise ValueError("GEMINI_MAX_OUTPUT_TOKENS must be at least 64")
    return value


def _config_for_model(model: str, *, temperature: float | None = 0.0, **extra: Any) -> Any:
    from google.genai import types
    kwargs: dict[str, Any] = {
        "max_output_tokens": _gemini_max_output_tokens(),
        "thinking_config": types.ThinkingConfig(thinking_level=_gemini_thinking_level()),
        **extra,
    }
    if temperature is not None and not model.startswith(("gemini-3.5", "gemini-3.6", "gemini-3.7")):
        kwargs["temperature"] = temperature
    return types.GenerateContentConfig(**kwargs)


def _generate_with_fallback(*, contents: Any, config_factory: Any, primary_model: str | None = None) -> tuple[Any, str]:
    client = _gemini_client()
    models = _gemini_models(primary_model)
    errors: list[str] = []
    quota_models: list[str] = []
    for model in models:
        for attempt in range(2):
            try:
                response = client.models.generate_content(model=model, contents=contents, config=config_factory(model))
                # The model actually used is returned to the caller. It used to
                # be written to os.environ, which threads overwrite.
                return response, model
            except Exception as exc:
                kind = _gemini_error_kind(exc)
                if kind == "quota":
                    quota_models.append(model)
                    errors.append(f"{model}: quota/resource exhausted")
                    break
                errors.append(f"{model} attempt {attempt + 1}: {exc}")
                if not _gemini_retryable(exc):
                    raise
                if attempt == 0:
                    time.sleep(1.0)
    if quota_models and len(quota_models) == len(models):
        raise RuntimeError("Gemini quota exhausted for all configured models; no provider retry is useful right now.")
    raise RuntimeError("All configured Gemini models failed after retries. " + " | ".join(errors))


def generate_text(messages: list[dict[str, str]], *, temperature: float = 0.0, model: str | None = None) -> str:
    contents: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        prefix = "System instructions:\n" if role == "system" else "User:\n" if role == "user" else f"{role.title()}:\n"
        contents.append(prefix + content)
    def factory(selected_model: str) -> Any:
        return _config_for_model(selected_model, temperature=temperature)
    try:
        response, _ = _generate_with_fallback(contents="\n\n".join(contents), config_factory=factory, primary_model=model)
    except Exception as exc:
        raise RuntimeError(f"Gemini generation failed: {exc}") from exc
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response")
    return text


def generate_with_tools(contents: Sequence[Any], *, declarations: Sequence[Any], system_instruction: str, temperature: float | None = 0.0, model: str | None = None) -> Any:
    from google.genai import types
    def factory(selected_model: str) -> Any:
        return _config_for_model(selected_model, temperature=temperature, system_instruction=system_instruction, tools=[types.Tool(function_declarations=list(declarations))], automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    response, _ = _generate_with_fallback(contents=list(contents), config_factory=factory, primary_model=model)
    return response


def generate_structured(contents: Any, *, system_instruction: str, response_schema: dict[str, Any], temperature: float | None = 0.0, model: str | None = None) -> str:
    from google.genai import types
    def factory(selected_model: str) -> Any:
        return _config_for_model(selected_model, temperature=temperature, system_instruction=system_instruction, automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True), response_mime_type="application/json", response_schema=response_schema)
    response, _ = _generate_with_fallback(contents=contents, config_factory=factory, primary_model=model)
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini structured generation returned an empty response")
    return text
