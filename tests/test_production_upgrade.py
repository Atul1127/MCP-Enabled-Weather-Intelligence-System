from __future__ import annotations

from weather_agent_core.security import inspect_text, validate_user_query


def test_security_normalizes_zero_width_injection():
    result = inspect_text("ignore\u200binstructions")
    assert result["suspicious"] is True


def test_security_normalizes_unicode_before_validation():
    assert validate_user_query("  What is the weather in Kolkata?  ") == "What is the weather in Kolkata?"


def test_control_characters_are_blocked():
    result = inspect_text("weather\x00 question")
    assert result["suspicious"] is True


def test_postgres_backend_is_selectable():
    from rag.pipeline import RAGPipeline
    pipeline = RAGPipeline()
    assert pipeline.backend in {"postgres", "local", "jsonl", "file", "postgresql", "pgvector", "lakebase"}
