"""Context compression adapter."""
from __future__ import annotations
from typing import Any
from advanced_rag import compress_context as _compress_context


def compress(query: str, documents: list[dict[str, Any]], max_chars: int | None = None):
    if max_chars is None:
        return _compress_context(query, documents)
    return _compress_context(query, documents, max_chars=max_chars)
