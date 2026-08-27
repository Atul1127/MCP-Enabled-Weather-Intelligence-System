"""HTTP-facing RAG service adapter.

The retrieval implementation lives under ``rag/``. This module is deliberately
thin so Flask remains an adapter rather than a second RAG implementation.
"""
from __future__ import annotations
import os
from typing import Any
from llm_provider import generate_text, model_name
from rag.pipeline import RAGPipeline

SYSTEM_PROMPT = """You are a grounded weather knowledge assistant. Use only the supplied retrieved evidence. Do not invent facts. Cite factual knowledge claims with [S1], [S2], etc. Clearly distinguish weather knowledge from live forecasts and official warnings. If evidence is insufficient, say so."""
_pipeline = RAGPipeline()

def answer_weather_question(query: str, top_k: int = 5, location: str | None = None, state: str | None = None) -> dict[str, Any]:
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty")
    top_k = max(1, min(20, int(top_k)))
    result = _pipeline.retrieve(query, location=location, state=state, top_k=top_k)
    answer = generate_text([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question:\n{query}\n\nRetrieved evidence:\n{result.context}"},
    ], temperature=0.0)
    answer, cited_sources = _pipeline.validate_answer(answer, result.sources)
    return {
        "success": True,
        "query": query,
        "intent": result.plan.intent,
        "answer": answer,
        "documents": result.documents,
        "context": result.context,
        "sources": cited_sources,
        "model": os.environ.get("GEMINI_LAST_MODEL", model_name()),
    }
