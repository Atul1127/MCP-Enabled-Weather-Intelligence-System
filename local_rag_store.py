"""Local file-backed retrieval store for zero-cost development.

The store and embedding model are process-level singletons. This avoids
rebuilding the BM25 index and dense corpus embeddings on every MCP request.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = os.environ.get(
    "WEATHER_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
CORPUS_PATH = Path(
    os.environ.get("WEATHER_LOCAL_RAG_CORPUS", "data/weather_knowledge.jsonl")
)

_model: SentenceTransformer | None = None
_store: "LocalRagStore | None" = None
_model_lock = Lock()
_store_lock = Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


class LocalRagStore:
    def __init__(self, path: Path = CORPUS_PATH) -> None:
        self.path = path
        self.rows = self._load(path)
        if not self.rows:
            raise RuntimeError(f"Local RAG corpus is empty: {path}")
        self.texts = [self._search_text(row) for row in self.rows]
        self.tokens = [self._tokenize(text) for text in self.texts]
        self.bm25 = BM25Okapi(self.tokens)
        self.embeddings = np.asarray(
            _get_model().encode(
                self.texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    @staticmethod
    def _load(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Local RAG corpus not found: {path}")
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    @staticmethod
    def _search_text(row: dict[str, Any]) -> str:
        fields = (
            "title", "location", "state", "source", "source_type",
            "topic", "date", "text",
        )
        return " ".join(str(row.get(key) or "") for key in fields)

    def filtered_rows(
        self,
        location: str | None = None,
        state: str | None = None,
        source_type: str | None = None,
    ) -> list[int]:
        def match(row: dict[str, Any]) -> bool:
            if location and str(row.get("location", "")).lower() != location.lower():
                return False
            if state and str(row.get("state", "")).lower() != state.lower():
                return False
            if source_type and str(row.get("source_type", "")).lower() != source_type.lower():
                return False
            return True
        return [i for i, row in enumerate(self.rows) if match(row)]

    def dense_search(self, query: str, limit: int, allowed: list[int] | None = None) -> list[dict[str, Any]]:
        vector = np.asarray(
            _get_model().encode([query], normalize_embeddings=True, show_progress_bar=False)[0],
            dtype=np.float32,
        )
        scores = self.embeddings @ vector
        indices = allowed if allowed is not None else list(range(len(self.rows)))
        ranked = sorted(indices, key=lambda i: float(scores[i]), reverse=True)[:limit]
        return [{**self.rows[i], "dense_score": float(scores[i])} for i in ranked]

    def bm25_search(self, query: str, limit: int, allowed: list[int] | None = None) -> list[dict[str, Any]]:
        scores = self.bm25.get_scores(self._tokenize(query))
        indices = allowed if allowed is not None else list(range(len(self.rows)))
        ranked = sorted(indices, key=lambda i: float(scores[i]), reverse=True)[:limit]
        return [
            {**self.rows[i], "bm25_score": float(scores[i])}
            for i in ranked
            if float(scores[i]) > 0
        ]


def get_store() -> LocalRagStore:
    """Return the process-wide RAG store, building it at most once."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = LocalRagStore()
    return _store
