from __future__ import annotations

import pytest

import services.llm as llm_provider
from services.llm import _generate_with_fallback, _gemini_error_kind, _gemini_retryable


class FakeQuotaError(Exception):
    status_code = 429

    def __str__(self) -> str:
        return "RESOURCE_EXHAUSTED: free tier quota exceeded"


class FakeUnavailableError(Exception):
    status_code = 503


def test_gemini_quota_errors_are_not_retried():
    exc = FakeQuotaError()
    assert _gemini_error_kind(exc) == "quota"
    assert _gemini_retryable(exc) is False


def test_gemini_transient_errors_are_retried():
    exc = FakeUnavailableError()
    assert _gemini_error_kind(exc) == "transient"
    assert _gemini_retryable(exc) is True


def test_gemini_permanent_errors_are_not_retried():
    exc = ValueError("invalid request")
    assert _gemini_error_kind(exc) == "permanent"
    assert _gemini_retryable(exc) is False


def test_quota_failure_skips_same_model_retry(monkeypatch: pytest.MonkeyPatch):
    class FakeModels:
        calls = 0

        def generate_content(self, **_: object) -> None:
            self.calls += 1
            raise FakeQuotaError()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(llm_provider, "_GEMINI_CLIENT", FakeClient())
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "")

    with pytest.raises(RuntimeError, match="quota exhausted"):
        _generate_with_fallback(
            contents="test",
            config_factory=lambda _: object(),
            primary_model="gemini-test",
        )

    assert FakeClient.models.calls == 1


def test_gemini_client_has_request_timeout(monkeypatch):
    import services.llm as llm
    captured = {}

    class FakeClient:
        def __init__(self, api_key, http_options=None):
            captured["timeout"] = http_options.timeout

    import google.genai as genai
    monkeypatch.setattr(genai, "Client", FakeClient)
    monkeypatch.setattr(llm, "_GEMINI_CLIENT", None)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "25")
    llm._gemini_client()
    assert captured["timeout"] == 25000


def test_timed_out_model_falls_back_without_retrying(monkeypatch):
    import services.llm as llm
    attempts = []

    class Models:
        def generate_content(self, model, contents, config):
            attempts.append(model)
            if model == "slow-model":
                raise TimeoutError("The read operation timed out")
            return "response"

    monkeypatch.setattr(llm, "_gemini_client", lambda: type("C", (), {"models": Models()})())
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "fast-model")
    response, model = llm._generate_with_fallback(contents="hi", config_factory=lambda m: None, primary_model="slow-model")
    assert (response, model) == ("response", "fast-model")
    assert attempts == ["slow-model", "fast-model"]  # no second wait on the slow model
