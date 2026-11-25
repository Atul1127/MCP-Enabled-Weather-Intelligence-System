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


def test_number_in_message_is_not_a_server_error():
    assert _gemini_error_kind(ValueError("prompt is 500 tokens over the limit")) == "permanent"


def test_status_name_is_used_when_present():
    class FakeApiError(Exception):
        code = 503
        status = "UNAVAILABLE"

    class FakeQuota(Exception):
        code = 429
        status = "RESOURCE_EXHAUSTED"

    assert _gemini_error_kind(FakeApiError("x")) == "transient"
    assert _gemini_error_kind(FakeQuota("x")) == "quota"


def test_model_name_is_not_written_to_the_environment(monkeypatch: pytest.MonkeyPatch):
    import os

    class FakeModels:
        def generate_content(self, **_: object) -> str:
            return "ok"

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(llm_provider, "_GEMINI_CLIENT", FakeClient())
    monkeypatch.delenv("GEMINI_LAST_MODEL", raising=False)
    response, model = _generate_with_fallback(contents="x", config_factory=lambda _: object(), primary_model="gemini-test")
    assert (response, model) == ("ok", "gemini-test")
    assert "GEMINI_LAST_MODEL" not in os.environ
