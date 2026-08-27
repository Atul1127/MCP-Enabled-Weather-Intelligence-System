"""Orchestration boundary for the weather RAG pipeline.

This module owns pipeline composition. Retrieval implementations, reranking,
compression and citation handling stay behind small adapters so each stage
can later be replaced or evaluated independently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from local_rag_store import get_store
from rag.query.analyzer import QueryPlan, analyze
from rag.retrieval.dense import search as dense_search
from rag.retrieval.sparse import search as sparse_search
from rag.retrieval.hybrid import fuse
from rag.reranking.cross_encoder import rerank
from rag.context.compressor import compress
from rag.citations.validator import validate


@dataclass
class RetrievalResult:
    plan: QueryPlan
    documents: list[dict[str, Any]]
    context: str
    sources: list[dict[str, Any]]


class RAGPipeline:
    """Composable retrieval pipeline with one public retrieve operation."""

    def __init__(self, *, dense_k: int = 30, sparse_k: int = 30, fusion_k: int = 5, top_k: int = 5):
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.fusion_k = fusion_k
        self.top_k = top_k

    def retrieve(self, query: str, *, location: str | None = None, state: str | None = None) -> RetrievalResult:
        plan = analyze(query, location=location, state=state)
        store = get_store()
        allowed = store.filtered_rows(location=location, state=state)

        # Query expansion remains optional at this boundary. The legacy
        # advanced_rag implementation is still the compatibility path while
        # the new stages are migrated one-by-one.
        variants = [plan.query]
        dense = dense_search(store, variants[0], self.dense_k, allowed)
        sparse = sparse_search(store, variants[0], self.sparse_k, allowed)
        fused = fuse(dense, sparse, self.fusion_k)
        ranked = rerank(plan.query, fused, min(self.top_k, len(fused)))
        context, sources = compress(plan.query, ranked)
        return RetrievalResult(plan, ranked, context, sources)

    @staticmethod
    def validate_answer(answer: str, sources: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        return validate(answer, sources)
