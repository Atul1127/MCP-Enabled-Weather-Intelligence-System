"""Simple context formatting for retrieved weather documents."""
from __future__ import annotations

import os
from typing import Any

MAX_CONTEXT_CHARS = int(os.environ.get("WEATHER_RAG_MAX_CONTEXT_CHARS", "9000"))


def _text(row: dict[str, Any]) -> str:
    return str(row.get("text") or row.get("narrative_text") or row.get("chunk_text") or row.get("headline") or "").strip()


def _id(row: dict[str, Any]) -> Any:
    return row.get("id", row.get("document_id"))


def compress(documents: list[dict[str, Any]], max_chars: int | None = None) -> tuple[str, list[dict[str, Any]]]:
    """Format top-k retrieved documents into bounded, citation-ready context."""
    limit = MAX_CONTEXT_CHARS if max_chars is None else max(1, int(max_chars))
    blocks: list[str] = []
    sources: list[dict[str, Any]] = []
    used = 0

    for i, row in enumerate(documents, 1):
        text = _text(row)
        if not text:
            continue
        block = (
            f"[S{i}] Topic={row.get('topic') or row.get('headline') or 'general'}; "
            f"Source={row.get('source') or row.get('source_type') or 'unknown'}; "
            f"Location={row.get('location') or 'general'}\n{text}"
        )
        remaining = limit - used
        if remaining <= 0:
            break
        if len(block) > remaining:
            if remaining < 80:
                break
            block = block[:remaining]

        blocks.append(block)
        used += len(block)
        sources.append({
            "citation": f"S{i}",
            "id": _id(row),
            "document_id": row.get("document_id", _id(row)),
            "title": row.get("title") or row.get("headline"),
            "source": row.get("source") or row.get("source_type"),
            "topic": row.get("topic") or row.get("headline"),
            "rrf_score": row.get("rrf_score"),
            "fusion_score": row.get("fusion_score"),
            "retrieval_confidence": row.get("retrieval_confidence"),
        })

    return "\n\n---\n\n".join(blocks), sources
