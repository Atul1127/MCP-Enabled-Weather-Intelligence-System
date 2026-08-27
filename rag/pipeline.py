"""Composable local-first RAG pipeline.

Query -> analysis -> optional multi-query expansion -> metadata filtering ->
dense/BM25 retrieval -> confidence-aware RRF -> cross-encoder reranking ->
diversity selection -> query-aware context compression.

Generation is intentionally outside retrieval: the agent synthesizer owns the
final grounded response across live MCP observations and RAG evidence.
"""
from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Any, Callable
from local_rag_store import get_store
from rag.query.analyzer import QueryPlan, analyze
from rag.query.expansion import expand
from rag.retrieval.dense import search as dense_search
from rag.retrieval.sparse import search as sparse_search
from rag.retrieval.hybrid import fuse
from rag.retrieval.diversity import select_mmr
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
    def __init__(self, *, dense_k: int = 30, sparse_k: int = 30, fusion_k: int = 10, top_k: int = 5, mmr_lambda: float = 0.75):
        if min(dense_k, sparse_k, fusion_k, top_k) < 1:
            raise ValueError("RAG retrieval limits must be positive")
        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be between 0 and 1")
        self.dense_k, self.sparse_k, self.fusion_k, self.top_k = dense_k, sparse_k, fusion_k, top_k
        self.mmr_lambda = mmr_lambda

    @staticmethod
    def _gemini_expand(query: str) -> str:
        from llm_provider import generate_text
        return generate_text([{
            "role": "user",
            "content": (
                'Rewrite the weather knowledge query into exactly two concise retrieval queries. '
                'Preserve every location, date, hazard, activity, and comparison term. '
                'Return JSON only as {"queries":["...","..."]}. Query: ' + query
            ),
        }], temperature=0.0)

    def retrieve(
        self,
        query: str,
        *,
        location: str | None = None,
        state: str | None = None,
        source_type: str | None = None,
        top_k: int | None = None,
        expand_query: Callable[[str], str] | None = None,
    ) -> RetrievalResult:
        text = query.strip()
        if not text:
            raise ValueError("Query cannot be empty")
        limit = self.top_k if top_k is None else max(1, min(20, int(top_k)))
        plan = analyze(text, location=location, state=state)
        store = get_store()
        allowed = store.filtered_rows(location=location, state=state, source_type=source_type)
        generator = expand_query
        if generator is None and plan.needs_expansion and os.environ.get("WEATHER_RAG_LLM_EXPANSION", "0") == "1":
            generator = self._gemini_expand
        variants = expand(plan.query, generator) if plan.needs_expansion else [plan.query]
        variants = list(dict.fromkeys([v.strip() for v in variants if v and v.strip()])) or [plan.query]
        dense_sets = [dense_search(store, q, self.dense_k, allowed) for q in variants]
        sparse_sets = [sparse_search(store, q, self.sparse_k, allowed) for q in variants]
        fused = fuse(
            [item for results in dense_sets for item in results],
            [item for results in sparse_sets for item in results],
            self.fusion_k,
        )
        rerank_candidates = fused[:max(limit, min(self.fusion_k, limit * 2))]
        ranked = rerank(plan.query, rerank_candidates, min(len(rerank_candidates), max(limit, 2)))
        selected = select_mmr(ranked, limit, lambda_mult=self.mmr_lambda)
        context, sources = compress(plan.query, selected)
        return RetrievalResult(plan, selected, context, sources)

    @staticmethod
    def validate_answer(answer: str, sources: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        return validate(answer, sources)
