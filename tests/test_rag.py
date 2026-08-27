import pytest

import rag_service
from rag.pipeline import RetrievalResult
from rag.query.analyzer import QueryPlan


def _result(documents, context, sources):
    plan = QueryPlan("test query", "knowledge", "simple", False, {"location": None, "state": None})
    return RetrievalResult(plan, documents, context, sources)


def test_answer_weather_question(monkeypatch):
    documents = [{"id": "doc1", "location": "Kolkata", "source": "weather-kb", "text": "Temperature will be 32 °C with a high chance of rain."}]
    sources = [{"citation": "S1", "id": "doc1", "document_id": "doc1", "source": "weather-kb"}]
    monkeypatch.setattr(rag_service._pipeline, "retrieve", lambda *args, **kwargs: _result(documents, "[S1] Location=Kolkata\nTemperature will be 32 °C with a high chance of rain.", sources))
    monkeypatch.setattr(rag_service, "generate_text", lambda *args, **kwargs: "Kolkata may receive rain tomorrow. [S1]")

    result = rag_service.answer_weather_question("Will it rain in Kolkata tomorrow?", top_k=5)

    assert result["success"] is True
    assert result["answer"].startswith("Kolkata may receive rain tomorrow.")
    assert result["documents"] == documents
    assert result["sources"] == sources
    assert result["model"]
    assert result["intent"] == "knowledge"


def test_empty_query_is_rejected():
    with pytest.raises(ValueError, match="Query cannot be empty"):
        rag_service.answer_weather_question("")


def test_top_k_is_clamped_and_applied(monkeypatch):
    documents = [{"id": str(i), "text": f"Document {i}"} for i in range(1, 7)]
    sources = [{"citation": f"S{i}", "id": str(i)} for i in range(1, 7)]
    monkeypatch.setattr(rag_service._pipeline, "retrieve", lambda *args, **kwargs: _result(documents, "context", sources))
    monkeypatch.setattr(rag_service, "generate_text", lambda *args, **kwargs: "Answer [S1]")

    result = rag_service.answer_weather_question("weather", top_k=2)
    assert len(result["documents"]) == 2


def test_no_documents_returns_empty_evidence(monkeypatch):
    monkeypatch.setattr(rag_service._pipeline, "retrieve", lambda *args, **kwargs: _result([], "", []))
    monkeypatch.setattr(rag_service, "generate_text", lambda *args, **kwargs: "Insufficient evidence.")

    result = rag_service.answer_weather_question("Unknown weather question")

    assert result["success"] is True
    assert result["documents"] == []
    assert result["sources"] == []
    assert result["answer"] == "Insufficient evidence."
