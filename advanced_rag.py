"""Advanced hybrid RAG for the weather knowledge base.

Pipeline:
query analysis -> multi-query expansion -> metadata filtering -> dense + BM25
-> RRF -> cross-encoder reranking -> context compression -> grounded answer.

Everything is local: embeddings/reranker run with sentence-transformers and the
LLM runs through Ollama.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

import lakebase
from observability import emit, new_trace_id, span

EMBEDDING_MODEL = os.environ.get("WEATHER_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.environ.get("WEATHER_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
LLM_MODEL = os.environ.get("WEATHER_LLM_MODEL", "llama3.2:3b")
RRF_K = int(os.environ.get("WEATHER_RRF_K", "60"))
DEFAULT_TOP_K = int(os.environ.get("WEATHER_RAG_TOP_K", "5"))
VECTOR_CANDIDATES = int(os.environ.get("WEATHER_VECTOR_CANDIDATES", "30"))
BM25_CANDIDATES = int(os.environ.get("WEATHER_BM25_CANDIDATES", "30"))
BM25_CORPUS_LIMIT = int(os.environ.get("WEATHER_BM25_CORPUS_LIMIT", "5000"))
RERANK_CANDIDATES = int(os.environ.get("WEATHER_RERANK_CANDIDATES", "20"))

_embedding_model: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None


def embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w.-]+", text.lower())


def _rows(location: str | None = None, state: str | None = None) -> list[dict[str, Any]]:
    clauses = []
    params: list[Any] = []
    if location:
        clauses.append("LOWER(d.location) = LOWER(%s)")
        params.append(location)
    if state:
        clauses.append("LOWER(d.state) = LOWER(%s)")
        params.append(state)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"""
        SELECT d.id AS document_id, d.location, d.state, d.district,
               d.source, d.source_type, d.headline, d.forecast_date,
               d.temperature_min_c, d.temperature_max_c, d.rainfall_mm,
               d.precipitation_probability, d.weather_code, d.severity,
               d.issued_at, e.chunk_index, e.chunk_text
        FROM weather_embeddings e
        JOIN weather_documents d ON d.id = e.document_id
        {where}
        ORDER BY d.forecast_date DESC NULLS LAST
        LIMIT %s
    """
    params.append(BM25_CORPUS_LIMIT)
    return lakebase.run_query(sql, tuple(params))


def expand_query(query: str, trace_id: str) -> list[str]:
    """Generate two cheap local query variants; fall back safely if Ollama fails."""
    prompt = (
        "Rewrite the weather search query into two short retrieval queries. "
        "Preserve location, dates, hazards and activity. Return JSON only as "
        "{\"queries\":[\"...\",\"...\"]}. Query: " + query
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


def dense_search(query: str, rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    vector = embedding_model().encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    literal = "[" + ",".join(repr(float(v)) for v in vector) + "]"
    ids = [r["document_id"] for r in rows]
    placeholders = ",".join(["%s"] * len(ids))
    sql = f"""
        SELECT d.id AS document_id, d.location, d.state, d.district, d.source,
               d.source_type, d.headline, d.forecast_date, d.temperature_min_c,
               d.temperature_max_c, d.rainfall_mm, d.precipitation_probability,
               d.weather_code, d.severity, d.issued_at, e.chunk_index, e.chunk_text,
               1 - (e.embedding <=> %s::vector) AS similarity
        FROM weather_embeddings e
        JOIN weather_documents d ON d.id = e.document_id
        WHERE d.id IN ({placeholders})
        ORDER BY e.embedding <=> %s::vector ASC
        LIMIT %s
    """
    params = [literal, *ids, literal, limit]
    return lakebase.run_query(sql, tuple(params))


def bm25_search(query: str, rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    corpus = []
    for row in rows:
        text = " ".join(str(row.get(k) or "") for k in (
            "location", "state", "district", "headline", "source_type", "severity", "chunk_text"
        ))
        corpus.append(tokenize(text))
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    result = []
    for i in ranked[:limit]:
        if scores[i] <= 0:
            continue
        item = dict(rows[i])
        item["bm25_score"] = float(scores[i])
        result.append(item)
    return result


def rrf_merge(result_sets: list[list[dict[str, Any]]], top_k: int) -> list[dict[str, Any]]:
    fused: dict[Any, dict[str, Any]] = {}
    for results in result_sets:
        for rank, row in enumerate(results, 1):
            key = row.get("document_id")
            if not key:
                continue
            item = fused.setdefault(key, {**row, "rrf_score": 0.0, "retrieval_ranks": []})
            item["rrf_score"] += 1.0 / (RRF_K + rank)
            item["retrieval_ranks"].append(rank)
            for k, v in row.items():
                if v is not None:
                    item.setdefault(k, v)
    return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)[:top_k]


def rerank(query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if not candidates:
        return []
    pairs = [(query, str(row.get("chunk_text") or "")) for row in candidates]
    scores = reranker().predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
    output = []
    for row, score in ranked[:top_k]:
        item = dict(row)
        item["reranker_score"] = float(score)
        output.append(item)
    return output


def build_context(documents: list[dict[str, Any]], max_chars: int = 12000) -> tuple[str, list[dict[str, Any]]]:
    blocks = []
    sources = []
    used = 0
    for i, row in enumerate(documents, 1):
        text = str(row.get("chunk_text") or "").strip()
        if not text:
            continue
        block = (
            f"[S{i}] Location={row.get('location')}; State={row.get('state')}; "
            f"Date={row.get('forecast_date')}; Source={row.get('source')}; "
            f"Severity={row.get('severity')}\n{text}"
        )
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining < 200:
                break
            block = block[:remaining]
        blocks.append(block)
        used += len(block)
        sources.append({
            "citation": f"S{i}",
            "document_id": row.get("document_id"),
            "source": row.get("source"),
            "location": row.get("location"),
            "forecast_date": row.get("forecast_date"),
            "rrf_score": row.get("rrf_score"),
            "reranker_score": row.get("reranker_score"),
        })
    return "\n\n---\n\n".join(blocks), sources


SYSTEM_PROMPT = """You are an Indian Weather Intelligence assistant.
Use only the supplied evidence. Never invent weather facts. Clearly distinguish
live/current conditions from forecasts and historical guidance. Cite factual
claims with [S1], [S2], etc. If evidence is insufficient, say so. Never call an
application-level forecast hazard an official government warning unless the
evidence explicitly identifies it as one. Keep the answer concise and actionable."""


def answer(query: str, top_k: int = DEFAULT_TOP_K, location: str | None = None, state: str | None = None) -> dict[str, Any]:
    trace_id = new_trace_id()
    started = time.perf_counter()
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    query = query.strip()
    emit("rag.start", trace_id=trace_id, query=query, top_k=top_k)
    try:
        with span("retrieve", trace_id=trace_id) as info:
            rows = _rows(location=location, state=state)
            variants = expand_query(query, trace_id)
            dense_sets = [dense_search(q, rows, VECTOR_CANDIDATES) for q in variants]
            sparse_sets = [bm25_search(q, rows, BM25_CANDIDATES) for q in variants]
            fused = rrf_merge(dense_sets + sparse_sets, RERANK_CANDIDATES)
            ranked = rerank(query, fused, top_k)
            info.update(corpus=len(rows), query_variants=len(variants), candidates=len(fused), returned=len(ranked))

        context, sources = build_context(ranked)
        if not context:
            return {"success": False, "query": query, "error": "No relevant evidence found.", "trace_id": trace_id}

        with span("generation", trace_id=trace_id, model=LLM_MODEL) as info:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question:\n{query}\n\nEvidence:\n{context}"},
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
                "strategy": "multi-query + dense + BM25 + RRF + cross-encoder",
                "candidates": len(fused),
                "returned": len(ranked),
            },
            "sources": sources,
            "trace_id": trace_id,
            "latency_ms": latency_ms,
        }
    except Exception as exc:
        emit("rag.error", trace_id=trace_id, error=str(exc))
        raise
