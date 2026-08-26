"""Advanced local-first hybrid RAG for the weather knowledge base.

Pipeline:
query analysis -> optional multi-query expansion -> metadata filtering -> dense + BM25
-> confidence-aware RRF -> adaptive cross-encoder reranking -> context compression
-> grounded Ollama answer.
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
CONFIDENCE_WEIGHT = float(os.environ.get("WEATHER_RRF_CONFIDENCE_WEIGHT", "0.5"))
DEFAULT_TOP_K = int(os.environ.get("WEATHER_RAG_TOP_K", "5"))
DENSE_CANDIDATES = int(os.environ.get("WEATHER_VECTOR_CANDIDATES", "30"))
BM25_CANDIDATES = int(os.environ.get("WEATHER_BM25_CANDIDATES", "30"))
RERANK_CANDIDATES = int(os.environ.get("WEATHER_RERANK_CANDIDATES", "5"))
RERANK_MIN_CANDIDATES = int(os.environ.get("WEATHER_RERANK_MIN_CANDIDATES", "2"))
MAX_CONTEXT_CHARS = int(os.environ.get("WEATHER_RAG_MAX_CONTEXT_CHARS", "9000"))

_reranker: CrossEncoder | None = None


def reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def _should_expand_query(query: str) -> bool:
    text = query.lower().strip()
    words = re.findall(r"\w+", text)
    if len(words) <= 10:
        return False
    return any(marker in text for marker in ("compare", "difference", "versus", " vs ", "why", "how does", "what factors", "relationship", "associated", "conditions", "typical", "forecast", "outdoor"))


def _parse_expansion(raw: str) -> list[str]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I | re.S).strip()
    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match and match.group(0) != cleaned:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        variants = data.get("queries", []) if isinstance(data, dict) else []
        if isinstance(variants, list):
            return [str(x).strip() for x in variants if str(x).strip()]
    return []


def expand_query(query: str, trace_id: str) -> list[str]:
    """Use LLM expansion only for complex queries; simple queries skip an LLM round."""
    if not _should_expand_query(query):
        emit("query_expansion.skipped", trace_id=trace_id, reason="simple_query")
        return [query]
    prompt = "Rewrite the weather search query into two short retrieval queries. Preserve location, date, hazard and activity terms. Return JSON only as {\"queries\":[\"...\",\"...\"]}. Query: " + query
    with span("query_expansion", trace_id=trace_id) as info:
        try:
            response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], options={"temperature": 0})
            variants = _parse_expansion(response["message"]["content"])
            if not variants:
                info.update(ok=False, fallback=True, error="invalid query-expansion JSON")
                emit("query_expansion.fallback", trace_id=trace_id, error="invalid query-expansion JSON")
                return [query]
            info["variants"] = len(variants[:2])
            return [query, *variants[:2]]
        except Exception as exc:
            info.update(ok=False, fallback=True, error=str(exc))
            emit("query_expansion.fallback", trace_id=trace_id, error=str(exc))
            return [query]


def _confidence(values: list[float], value: float) -> float:
    if not values: return 0.0
    lo, hi = min(values), max(values)
    if hi <= lo: return 1.0 if value > 0 else 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def confidence_aware_rrf(result_sets: list[tuple[str, list[dict[str, Any]]]], top_k: int) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    for channel, results in result_sets:
        score_key = "dense_score" if channel == "dense" else "bm25_score"
        values = [float(row.get(score_key, 0.0)) for row in results]
        for rank, row in enumerate(results, 1):
            key = str(row.get("id"))
            if not key or key == "None": continue
            confidence = _confidence(values, float(row.get(score_key, 0.0)))
            item = fused.setdefault(key, {**row, "rrf_score": 0.0, "confidence_score": 0.0, "retrieval_ranks": [], "retrieval_channels": []})
            item["rrf_score"] += 1.0 / (RRF_K + rank)
            item["confidence_score"] += CONFIDENCE_WEIGHT * confidence / (RRF_K + rank)
            item["retrieval_ranks"].append(rank); item["retrieval_channels"].append(channel)
            item["retrieval_confidence"] = max(float(item.get("retrieval_confidence", 0.0)), confidence)
    for item in fused.values(): item["fusion_score"] = item["rrf_score"] + item["confidence_score"]
    return sorted(fused.values(), key=lambda x: x["fusion_score"], reverse=True)[:top_k]


def rrf_merge(result_sets: list[list[dict[str, Any]]], top_k: int) -> list[dict[str, Any]]:
    typed = [("dense" if i % 2 == 0 else "bm25", results) for i, results in enumerate(result_sets)]
    return confidence_aware_rrf(typed, top_k)


def rerank(query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if not candidates: return []
    if len(candidates) < RERANK_MIN_CANDIDATES:
        return [{**row, "reranker_score": None} for row in candidates[:top_k]]
    pairs = [(query, str(row.get("text") or row.get("narrative_text") or "")) for row in candidates]
    scores = reranker().predict(pairs, batch_size=min(8, len(pairs)), show_progress_bar=False)
    ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
    return [{**row, "reranker_score": float(score)} for row, score in ranked[:top_k]]


def _query_terms(query: str) -> set[str]: return set(re.findall(r"[a-zA-Z0-9]+", query.lower()))


def compress_context(query: str, documents: list[dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> tuple[str, list[dict[str, Any]]]:
    terms = _query_terms(query); blocks: list[str] = []; sources: list[dict[str, Any]] = []; used = 0
    for i, row in enumerate(documents, 1):
        text = str(row.get("text") or row.get("narrative_text") or "").strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        ranked_sentences = sorted(sentences, key=lambda s: sum(1 for token in _query_terms(s) if token in terms), reverse=True)
        selected = " ".join(ranked_sentences[:4]) or text
        block = f"[S{i}] Topic={row.get('topic')}; Source={row.get('source')}; Location={row.get('location') or 'general'}\n{selected}"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining < 250: break
            block = block[:remaining]
        blocks.append(block); used += len(block)
        sources.append({"citation": f"S{i}", "id": row.get("id"), "title": row.get("title"), "source": row.get("source"), "topic": row.get("topic"), "rrf_score": row.get("rrf_score"), "fusion_score": row.get("fusion_score"), "retrieval_confidence": row.get("retrieval_confidence"), "reranker_score": row.get("reranker_score")})
    return "\n\n---\n\n".join(blocks), sources


def _ensure_citations(answer_text: str, sources: list[dict[str, Any]]) -> str:
    if not sources: return answer_text.strip()
    valid = [str(source.get("citation")) for source in sources if source.get("citation")]
    existing = set(re.findall(r"\[(S\d+)\]", answer_text or "")); missing = [citation for citation in valid if citation not in existing]
    text = (answer_text or "").strip()
    return text + ("\n\nSources: " + ", ".join(f"[{citation}]" for citation in missing) if missing else "")


SYSTEM_PROMPT = """You are an Indian Weather Intelligence assistant.
Use ONLY the supplied retrieved evidence. Never invent retrieved facts.
Cite factual claims with [S1], [S2], etc. If the evidence does not support a
claim, say that it is not established by the retrieved sources. Distinguish
reference guidance from live observations, forecasts, and official warnings.
Keep the answer concise and useful."""


def answer(query: str, top_k: int = DEFAULT_TOP_K, location: str | None = None, state: str | None = None) -> dict[str, Any]:
    trace_id = os.environ.get("WEATHER_TRACE_ID") or new_trace_id(); started = time.perf_counter()
    if not query or not query.strip(): raise ValueError("Query cannot be empty")
    query = query.strip(); emit("rag.start", trace_id=trace_id, query=query, top_k=top_k, backend="local")
    try:
        store = get_store(); allowed = store.filtered_rows(location=location, state=state); variants = expand_query(query, trace_id)
        with span("retrieval", trace_id=trace_id) as info:
            dense_sets = [store.dense_search(q, DENSE_CANDIDATES, allowed) for q in variants]
            sparse_sets = [store.bm25_search(q, BM25_CANDIDATES, allowed) for q in variants]
            typed_sets = [("dense", rows) for rows in dense_sets] + [("bm25", rows) for rows in sparse_sets]
            fused = confidence_aware_rrf(typed_sets, RERANK_CANDIDATES)
            info.update(backend="local", corpus=len(store.rows), filtered=len(allowed), query_variants=len(variants), dense_candidates=sum(len(x) for x in dense_sets), bm25_candidates=sum(len(x) for x in sparse_sets), fused_candidates=len(fused), fusion="confidence-aware-rrf")
        with span("reranking", trace_id=trace_id) as info:
            ranked = rerank(query, fused, min(top_k, len(fused))); info["candidates"] = len(fused); info["returned"] = len(ranked); info["model"] = RERANKER_MODEL if len(fused) >= RERANK_MIN_CANDIDATES else "skipped-small-candidate-set"
        with span("context_compression", trace_id=trace_id) as info:
            context, sources = compress_context(query, ranked); info["sources"] = len(sources); info["context_chars"] = len(context)
        if not context:
            emit("rag.no_evidence", trace_id=trace_id); return {"success": False, "query": query, "error": "No relevant evidence found in the local weather knowledge base.", "trace_id": trace_id}
        with span("generation", trace_id=trace_id, model=LLM_MODEL) as info:
            response = ollama.chat(model=LLM_MODEL, messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Question:\n{query}\n\nRetrieved evidence:\n{context}"}], options={"temperature": 0})
            text = response["message"]["content"].strip(); info["answer_chars"] = len(text)
        text = _ensure_citations(text, sources); latency_ms = round((time.perf_counter() - started) * 1000, 2); emit("rag.end", trace_id=trace_id, latency_ms=latency_ms, source_count=len(sources))
        return {"success": True, "query": query, "answer": text, "retrieval": {"backend": "local", "strategy": "optional multi-query + dense + BM25 + confidence-aware RRF + adaptive cross-encoder + compression", "corpus": len(store.rows), "filtered": len(allowed), "query_variants": len(variants), "fused_candidates": len(fused), "returned": len(ranked)}, "sources": sources, "trace_id": trace_id, "latency_ms": latency_ms}
    except Exception as exc:
        emit("rag.error", trace_id=trace_id, error=str(exc)); return {"success": False, "query": query, "error": str(exc), "trace_id": trace_id}
