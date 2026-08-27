import llm_provider


def test_provider_is_lazy_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    provider = llm_provider.GeminiProvider(model="gemini-test")
    assert provider.model == "gemini-test"
    assert provider._client is None


def test_model_fallback_order_is_deduplicated(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "fallback-a,primary,fallback-b")
    assert llm_provider._gemini_models() == ["primary", "fallback-a", "fallback-b"]


def test_retryable_errors_include_quota_and_transient_failures():
    assert llm_provider._gemini_retryable(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert llm_provider._gemini_retryable(RuntimeError("503 UNAVAILABLE"))
    assert llm_provider._gemini_retryable(RuntimeError("504 DEADLINE_EXCEEDED"))
    assert not llm_provider._gemini_retryable(RuntimeError("400 INVALID_ARGUMENT"))
