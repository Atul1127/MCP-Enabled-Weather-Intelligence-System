"""Simple RAG pipeline with local and PostgreSQL backends."""
from __future__ import annotations

from dataclasses import dataclass
import os
from threading import Lock
from typing import Any

from rag.query.analyzer import QueryPlan, analyze
from rag.retrieval.dense import search as dense_search
from rag.retrieval.sparse import search as sparse_search
from rag.retrieval.hybrid import fuse
from rag.context.compressor import compress
from rag.citations.validator import validate
from weather_agent_core.security import validate_location, validate_user_query


@dataclass
class RetrievalResult:
    plan: QueryPlan
    documents: list[dict[str, Any]]
    context: str
    sources: list[dict[str, Any]]


_EMBEDDING_MODEL: Any | None = None
_EMBEDDING_LOCK = Lock()


def _embedding_model() -> Any:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        with _EMBEDDING_LOCK:
            if _EMBEDDING_MODEL is None:
                name = os.environ.get("WEATHER_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
                _EMBEDDING_MODEL = SentenceTransformer(name)
    return _EMBEDDING_MODEL


class RAGPipeline:
    def __init__(self, *, dense_k: int = 30, sparse_k: int = 30, fusion_k: int = 10, top_k: int = 5):
        if min(dense_k, sparse_k, fusion_k, top_k) < 1:
            raise ValueError("RAG retrieval limits must be positive")
        self.dense_k, self.sparse_k, self.fusion_k, self.top_k = dense_k, sparse_k, fusion_k, top_k
        self.backend = os.environ.get("WEATHER_RAG_BACKEND", "postgres").strip().lower()
        self.dense_enabled = os.environ.get("WEATHER_RAG_DENSE", "0").strip().lower()
        self._store: Any | None = None

    def _get_store(self) -> Any:
        if self._store is None:
            if self.backend in {"local", "jsonl", "file"}:
                from rag.local_rag_store import get_store
            elif self.backend in {"postgres", "postgresql", "pgvector", "lakebase"}:
                from rag.postgres_store import get_store
            else:
                raise ValueError("WEATHER_RAG_BACKEND must be postgres or local")
            self._store = get_store()
        return self._store

    @staticmethod
    def _query_embedding(query: str) -> list[float]:
        vector = _embedding_model().encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        return [float(value) for value in vector]

    def _should_use_dense(self) -> bool:
        return self.dense_enabled in {"1", "true", "on", "yes"}

    def retrieve(
        self,
        query: str,
        *,
        location: str | None = None,
        state: str | None = None,
        source_type: str | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        text = validate_user_query(query)
        if location is not None:
            location = validate_location(location)
        if state is not None:
            state = validate_location(state)
        limit = self.top_k if top_k is None else max(1, min(20, int(top_k)))
        plan = analyze(text, location=location, state=state)
        store = self._get_store()
        allowed = store.filtered_rows(location=location, state=state, source_type=source_type)

        sparse = sparse_search(store, plan.query, self.sparse_k, allowed)
        if self._should_use_dense():
            if self.backend in {"local", "jsonl", "file"}:
                dense = dense_search(store, plan.query, self.dense_k, allowed)
            else:
                dense = store.dense_search(self._query_embedding(plan.query), self.dense_k, allowed)
            documents = fuse(dense, sparse, self.fusion_k)[:limit]
        else:
            documents = sparse[:limit]

        context, sources = compress(documents)
        return RetrievalResult(plan, documents, context, sources)

    @staticmethod
    def validate_answer(answer: str, sources: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        return validate(answer, sources)
