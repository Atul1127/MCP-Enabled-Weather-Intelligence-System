"""HTTP-facing RAG service adapter.

Retrieval lives under ``rag/`` and generation uses the Gemini provider. This
module owns request validation and HTTP-facing response shaping.
"""
from __future__ import annotations

import os
from typing import Any

from llm_provider import generate_text, model_name
from rag.pipeline import RAGPipeline, RetrievalResult
from weather_agent_core.security import validate_location, validate_top_k, validate_user_query

SYSTEM_PROMPT = """You are a grounded weather knowledge assistant. Use only the supplied retrieved evidence. Do not invent facts. Cite factual knowledge claims with [S1], [S2], etc. Clearly distinguish weather knowledge from live forecasts and official warnings. If evidence is insufficient, say so."""
_pipeline = RAGPipeline()


def _limit_result(result: RetrievalResult, top_k: int) -> RetrievalResult:
    return RetrievalResult(result.plan, list(result.documents[:top_k]), result.context, list(result.sources[:top_k]))


def answer_weather_question(query: str, top_k: int = 5, location: str | None = None, state: str | None = None) -> dict[str, Any]:
    query = validate_user_query(query)
    top_k = validate_top_k(top_k)
    if location is not None:
        location = validate_location(location)
    if state is not None:
        state = validate_location(state)
    result = _limit_result(_pipeline.retrieve(query, location=location, state=state, top_k=top_k), top_k)
    answer = generate_text([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question:\n{query}\n\nRetrieved evidence:\n{result.context}"},
    ], temperature=0.0)
    answer, cited_sources = _pipeline.validate_answer(answer, result.sources)
    return {"success": True, "query": query, "intent": result.plan.intent, "answer": answer, "documents": result.documents, "context": result.context, "sources": cited_sources, "model": os.environ.get("GEMINI_LAST_MODEL", model_name())}
