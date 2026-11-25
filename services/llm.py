"""Single Gemini gateway for text, tool-calling, and structured generation."""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Sequence

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass

_GEMINI_CLIENT: Any | None = None
logger = logging.getLogger("weather-rag.llm")


def _gemini_timeout_ms() -> int:
    """Per-request timeout; the SDK default is no timeout at all."""
    seconds = float(os.environ.get("GEMINI_TIMEOUT_SECONDS", "25"))
    if seconds <= 0:
        raise ValueError("GEMINI_TIMEOUT_SECONDS must be positive")
    return int(seconds * 1000)


def model_name() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")


def _gemini_models(primary: str | None = None) -> list[str]:
    primary_name = primary or model_name()
    configured = os.environ.get("GEMINI_FALLBACK_MODELS", "gemini-3.6-flash,gemini-3.5-flash-lite")
    return list(dict.fromkeys([primary_name] + [item.strip() for item in configured.split(",") if item.strip()]))


def _gemini_client() -> Any:
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in the process environment")
        from google.genai import types
        # Without a timeout one slow response can consume the whole request
        # budget before the fallback model is ever tried.
        _GEMINI_CLIENT = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=_gemini_timeout_ms()))
    return _GEMINI_CLIENT


def _gemini_error_kind(exc: Exception) -> str:
    text = str(exc).upper()
    if isinstance(exc, TimeoutError) or "TIMEOUT" in type(exc).__name__.upper() or "TIMED OUT" in text:
        return "timeout"
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)
    if "RESOURCE_EXHAUSTED" in text or "QUOTA_EXCEEDED" in text or "QUOTA EXHAUSTED" in text:
        return "quota"
    if status == 429 or any(marker in text for marker in ("429", "RATE LIMIT", "TOO MANY REQUESTS")):
        return "rate_limit"
    if status in {500, 502, 503, 504} or any(marker in text for marker in ("500", "502", "503", "504", "UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED")):
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
    # Gemini 3+/4 models are tuned for their default temperature; forcing 0.0 can
    # cause looping or degraded reasoning, so only older models get an override.
    if temperature is not None and not model.startswith(("gemini-3", "gemini-4")):
        kwargs["temperature"] = temperature
    return types.GenerateContentConfig(**kwargs)


def _generate_with_fallback(*, contents: Any, config_factory: Any, primary_model: str | None = None) -> tuple[Any, str]:
    client = _gemini_client()
    models = _gemini_models(primary_model)
    errors: list[str] = []
    quota_models: list[str] = []
    for model in models:
        for attempt in range(2):
            started = time.perf_counter()
            try:
                response = client.models.generate_content(model=model, contents=contents, config=config_factory(model))
                logger.info("Gemini %s answered in %.1fs (attempt %d)", model, time.perf_counter() - started, attempt + 1)
                os.environ["GEMINI_LAST_MODEL"] = model
                return response, model
            except Exception as exc:
                kind = _gemini_error_kind(exc)
                logger.warning("Gemini %s failed after %.1fs (%s): %s", model, time.perf_counter() - started, kind, str(exc)[:200])
                if kind == "timeout":
                    # Waiting the full timeout again on the same model rarely helps;
                    # move straight to the next configured model.
                    errors.append(f"{model}: timed out")
                    break
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
