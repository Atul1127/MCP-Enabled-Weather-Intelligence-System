from __future__ import annotations

from llm_provider import _gemini_error_kind, _gemini_retryable


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
