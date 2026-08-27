"""Query-aware context compression implementation."""
from __future__ import annotations
import os
import re
from typing import Any

MAX_CONTEXT_CHARS = int(os.environ.get("WEATHER_RAG_MAX_CONTEXT_CHARS", "9000"))


def _query_terms(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def compress(query: str, documents: list[dict[str, Any]], max_chars: int | None = None) -> tuple[str, list[dict[str, Any]]]:
    limit = max_chars or MAX_CONTEXT_CHARS
    terms = _query_terms(query)
    blocks: list[str] = []
    sources: list[dict[str, Any]] = []
    used = 0
    for i, row in enumerate(documents, 1):
        text = str(row.get("text") or row.get("narrative_text") or "").strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        ranked = sorted(sentences, key=lambda s: sum(token in terms for token in _query_terms(s)), reverse=True)
        selected = " ".join(ranked[:4]) or text
        block = f"[S{i}] Topic={row.get('topic')}; Source={row.get('source')}; Location={row.get('location') or 'general'}\n{selected}"
        if used + len(block) > limit:
            remaining = limit - used
            if remaining < 250:
                break
            block = block[:remaining]
        blocks.append(block)
        used += len(block)
        sources.append({
            "citation": f"S{i}", "id": row.get("id"), "title": row.get("title"),
            "source": row.get("source"), "topic": row.get("topic"),
            "rrf_score": row.get("rrf_score"), "fusion_score": row.get("fusion_score"),
            "retrieval_confidence": row.get("retrieval_confidence"), "reranker_score": row.get("reranker_score"),
        })
    return "\n\n---\n\n".join(blocks), sources
