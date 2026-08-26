"""Advanced local-first hybrid RAG for the weather knowledge base.

Pipeline:
query analysis -> multi-query expansion -> metadata filtering -> dense + BM25
-> RRF -> cross-encoder reranking -> context compression -> grounded Ollama answer.

The default backend is file-backed and requires no PostgreSQL, Lakebase, API key,
or paid service. A production database adapter can be selected explicitly with
WEATHER_RAG_BACKEND=postgres after the local pipeline is validated.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import ollama
from sentence_transformers import CrossEncoder

from local_rag_store import get_store
from observability import emit, new_trace_id, span

LLM_MODEL = os.environ.get("WEATHER_LLM_MODEL", "llama3.2:3b")
RERANKER_MODEL = os.environ.get("WEATHER_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RRF_K = int(os.environ.get("WEATHER_RRF_K", "60"))
DEFAULT_TOP_K = int(os.environ.get("WEATHER_RAG_TOP_K", "5"))
DENSE_CANDIDATES = int(os.environ.get("WEATHER_VECTOR_CANDIDATES", "30"))
BM25_CANDIDATES = int(os.environ.get("WEATHER_BM25_CANDIDATES", "30"))
RERANK_CANDIDATES = int(os.environ.get("WEATHER_RERANK_CANDIDATES", "20"))
MAX_CONTEXT_CHARS = int(os.environ.get("WEATHER_RAG_MAX_CONTEXT_CHARS", "9000"))

_reranker: CrossEncoder | None = None


def reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def expand_query(query: str, trace_id: str) -> list[str]:
    """Generate two local query variants; never make retrieval depend on expansion."""
    prompt = (
        "Rewrite the weather search query into two short retrieval queries. "
        "Preserve location, date, hazard and activity terms. Return JSON only "
        "as {\"queries\":[\"...\",\"...\"]}. Query: " + query
    )
    try:
        with span("query_expansion", trace_id=trace_id) as info:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )
            raw = response["message"]["content"].strip()
            data = json.loads(raw)
            variants = [str(x).strip() for x in data.get("queries", []) if str(x).strip()]
            info["variants"] = len(variants)
            return [query, *variants[:2]]
    except Exception as exc:
        emit("query_expansion.fallback", trace_id=trace_id, error=str(exc))
        return [query]


def rrf_merge(result_sets: list[list[dict[str, Any]]], top_k: int) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    for results in result_sets:
        for rank, row in enumerate(results, 1):
            key = str(row.get("id"))
            if not key or key == "None":
                continue
            item = fused.setdefault(key, {**row, "rrf_score": 0.0, "retrieval_ranks": []})
            item["rrf_score"] += 1.0 / (RRF_K + rank)
            item["retrieval_ranks"].append(rank)
    return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)[:top_k]


def rerank(query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if not candidates:
        return []
    pairs = [(query, str(row.get("text") or row.get("narrative_text") or "")) for row in candidates]
    scores = reranker().predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
    return [{**row, "reranker_score": float(score)} for row, score in ranked[:top_k]]


def _query_terms(query: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", query.lower()))


def compress_context(query: str, documents: list[dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> tuple[str, list[dict[str, Any]]]:
    """Compress retrieved documents by retaining the most query-relevant sentences."""
    terms = _query_terms(query)
    blocks: list[str] = []
    sources: list[dict[str, Any]] = []
    used = 0
    for i, row in enumerate(documents, 1):
        text = str(row.get("text") or row.get("narrative_text") or "").strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        ranked_sentences = sorted(
            sentences,
            key=lambda s: sum(1 for token in _query_terms(s) if token in terms),
            reverse=True,
        )
        selected = " ".join(ranked_sentences[:4]) or text
        block = (
            f"[S{i}] Topic={row.get('topic')}; Source={row.get('source')}; "
            f"Location={row.get('location') or 'general'}\n{selected}"
        )
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining < 250:
                break
            block = block[:remaining]
        blocks.append(block)
        used += len(block)
        sources.append({
            "citation": f"S{i}",
            "id": row.get("id"),
            "title": row.get("title"),
            "source": row.get("source"),
            "topic": row.get("topic"),
            "rrf_score": row.get("rrf_score"),
            "reranker_score": row.get("reranker_score"),
        })
    return "\n\n---\n\n".join(blocks), sources


SYSTEM_PROMPT = """You are an Indian Weather Intelligence assistant.
Use ONLY the supplied retrieved evidence. Never invent retrieved facts.
Cite factual claims with [S1], [S2], etc. If the evidence does not support a
claim, say that it is not established by the retrieved sources. Distinguish
reference guidance from live observations, forecasts, and official warnings.
Keep the answer concise and useful."""


def answer(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    location: str | None = None,
    state: str | None = None,
) -> dict[str, Any]:
    trace_id = new_trace_id()
    started = time.perf_counter()
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    query = query.strip()
    emit("rag.start", trace_id=trace_id, query=query, top_k=top_k, backend="local")
    try:
        store = get_store()
        allowed = store.filtered_rows(location=location, state=state)
        variants = expand_query(query, trace_id)
        with span("retrieval", trace_id=trace_id) as info:
            dense_sets = [store.dense_search(q, DENSE_CANDIDATES, allowed) for q in variants]
            sparse_sets = [store.bm25_search(q, BM25_CANDIDATES, allowed) for q in variants]
            fused = rrf_merge(dense_sets + sparse_sets, RERANK_CANDIDATES)
            info.update(
                backend="local",
                corpus=len(store.rows),
                filtered=len(allowed),
                query_variants=len(variants),
                dense_candidates=sum(len(x) for x in dense_sets),
                bm25_candidates=sum(len(x) for x in sparse_sets),
                fused_candidates=len(fused),
            )

        ranked = rerank(query, fused, top_k)
        context, sources = compress_context(query, ranked)
        if not context:
            emit("rag.no_evidence", trace_id=trace_id)
            return {
                "success": False,
                "query": query,
                "error": "No relevant evidence found in the local weather knowledge base.",
                "trace_id": trace_id,
                "retrieval": {"strategy": "multi-query + dense + BM25 + RRF + cross-encoder + compression"},
            }

        with span("generation", trace_id=trace_id, model=LLM_MODEL) as info:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question:\n{query}\n\nRetrieved evidence:\n{context}"},
                ],
                options={"temperature": 0},
            )
            text = response["message"]["content"].strip()
            info["answer_chars"] = len(text)

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        emit("rag.end", trace_id=trace_id, latency_ms=latency_ms, source_count=len(sources))
        return {
            "success": True,
            "query": query,
            "answer": text,
            "retrieval": {
                "backend": "local",
                "strategy": "multi-query + dense + BM25 + RRF + cross-encoder + compression",
                "corpus": len(store.rows),
                "filtered": len(allowed),
                "query_variants": len(variants),
                "fused_candidates": len(fused),
                "returned": len(ranked),
            },
            "sources": sources,
            "trace_id": trace_id,
            "latency_ms": latency_ms,
        }
    except Exception as exc:
        emit("rag.error", trace_id=trace_id, error=str(exc))
        return {"success": False, "query": query, "error": str(exc), "trace_id": trace_id}
