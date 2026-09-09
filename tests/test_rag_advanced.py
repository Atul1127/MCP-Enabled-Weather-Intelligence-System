from rag.context.compressor import compress


def test_context_formatting_preserves_documents_and_citations():
    documents = [
        {
            "id": "a",
            "text": "Heavy rainfall is common during the Indian monsoon.",
            "topic": "rainfall",
            "source": "local-weather-guide",
            "location": "India",
        },
        {
            "id": "b",
            "text": "Strong winds can create hazardous outdoor conditions.",
            "topic": "wind",
            "source": "local-weather-guide",
            "location": "India",
        },
    ]
    context, sources = compress(documents)
    assert "[S1]" in context
    assert "[S2]" in context
    assert "Heavy rainfall" in context
    assert [source["citation"] for source in sources] == ["S1", "S2"]


def test_context_formatting_enforces_limit():
    documents = [{"id": "a", "text": "x" * 1000}]
    context, sources = compress(documents, max_chars=100)
    assert len(context) <= 100
    assert sources
