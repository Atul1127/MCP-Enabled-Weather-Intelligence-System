"""Cross-encoder reranking with a cheap small-corpus fast path."""
from __future__ import annotations

import os
from threading import Lock
from typing import Any

from sentence_transformers import CrossEncoder

MODEL = os.environ.get("WEATHER_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# The bundled weather KB is intentionally small. Loading a second transformer
# for <= 10 candidates costs more than it improves ranking. Keep the reranker
# available for larger deployments by raising fusion_k/top_k or overriding this.
MIN_CANDIDATES = int(os.environ.get("WEATHER_RERANK_MIN_CANDIDATES", "11"))
_model: CrossEncoder | None = None
_model_lock = Lock()


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = CrossEncoder(MODEL)
    return _model


def rerank(query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if not candidates:
        return []
    if len(candidates) < MIN_CANDIDATES:
        return [{**row, "reranker_score": None} for row in candidates[:top_k]]
    pairs = [(query, str(row.get("text") or row.get("narrative_text") or "")) for row in candidates]
    scores = _get_model().predict(pairs, batch_size=min(8, len(pairs)), show_progress_bar=False)
    ranked = sorted(zip(candidates, scores), key=lambda item: float(item[1]), reverse=True)
    return [{**row, "reranker_score": float(score)} for row, score in ranked[:top_k]]
