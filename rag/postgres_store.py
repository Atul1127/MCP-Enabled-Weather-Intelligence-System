"""PostgreSQL + pgvector retrieval backend.

Unlike the local JSONL store, this backend keeps document vectors in PostgreSQL
and performs filtering, lexical ranking, and vector similarity in the database.
It is intended for multi-worker production deployments.
"""
from __future__ import annotations

from typing import Any

import lakebase


class PostgresRagStore:
    """Database-backed RAG store using PostgreSQL full-text search + pgvector."""

    def __init__(self) -> None:
        self._ensure_ready()

    @staticmethod
    def _ensure_ready() -> None:
        """Fail fast with a useful message when the vector schema is absent."""
        rows = lakebase.run_query(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'weather_documents'
            ) AS documents,
            EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'weather_embeddings'
            ) AS embeddings
            """
        )
        if not rows or not rows[0]["documents"] or not rows[0]["embeddings"]:
            raise RuntimeError(
                "RAG database schema is not initialized. Run "
                "lakebase.ensure_weather_tables() and the embedding indexer first."
            )

    @staticmethod
    def _where(location: str | None, state: str | None, source_type: str | None) -> tuple[str, list[Any]]:
        clauses = ["1 = 1"]
        params: list[Any] = []
        if location:
            clauses.append("d.location ILIKE %s")
            params.append(location)
        if state:
            clauses.append("d.state ILIKE %s")
            params.append(state)
        if source_type:
            clauses.append("d.source_type = %s")
            params.append(source_type)
        return " AND ".join(clauses), params

    def filtered_rows(self, location: str | None = None, state: str | None = None, source_type: str | None = None) -> list[str]:
        where, params = self._where(location, state, source_type)
        rows = lakebase.run_query(
            f"SELECT d.id FROM weather_documents d WHERE {where}",
            tuple(params),
        )
        return [str(row["id"]) for row in rows]

    def _ids_clause(self, allowed: list[str] | None) -> tuple[str, list[Any]]:
        if not allowed:
            return "", []
        placeholders = ",".join(["%s"] * len(allowed))
        return f" AND d.id IN ({placeholders})", list(allowed)

    def bm25_search(self, query: str, limit: int, allowed: list[str] | None = None) -> list[dict[str, Any]]:
        """Lexical retrieval using PostgreSQL's built-in full-text index."""
        extra, extra_params = self._ids_clause(allowed)
        rows = lakebase.run_query(
            f"""
            SELECT d.id, d.location, d.state, d.district, d.source, d.source_type,
                   d.headline, d.narrative_text, d.forecast_date,
                   d.temperature_min_c, d.temperature_max_c, d.rainfall_mm,
                   d.precipitation_probability, d.weather_code, d.severity,
                   ts_rank_cd(
                       to_tsvector('simple', concat_ws(' ', d.location, d.state,
                           d.district, d.headline, d.narrative_text)),
                       plainto_tsquery('simple', %s)
                   ) AS bm25_score
            FROM weather_documents d
            WHERE 1 = 1 {extra}
            ORDER BY bm25_score DESC, d.synced_at DESC NULLS LAST
            LIMIT %s
            """,
            tuple([query, *extra_params, int(limit)]),
        )
        return [dict(row) for row in rows]

    def dense_search(self, query_vector: list[float], limit: int, allowed: list[str] | None = None) -> list[dict[str, Any]]:
        """Vector retrieval against persisted pgvector embeddings."""
        vector_literal = "[" + ",".join(repr(float(v)) for v in query_vector) + "]"
        extra, extra_params = self._ids_clause(allowed)
        rows = lakebase.run_query(
            f"""
            SELECT d.id, d.location, d.state, d.district, d.source, d.source_type,
                   d.headline, d.narrative_text, d.forecast_date,
                   d.temperature_min_c, d.temperature_max_c, d.rainfall_mm,
                   d.precipitation_probability, d.weather_code, d.severity,
                   e.chunk_text,
                   1 - (e.embedding <=> %s::vector) AS dense_score
            FROM weather_embeddings e
            JOIN weather_documents d ON d.id = e.document_id
            WHERE 1 = 1 {extra}
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
            """,
            tuple([vector_literal, *extra_params, vector_literal, int(limit)]),
        )
        return [dict(row) for row in rows]


def get_store() -> PostgresRagStore:
    return PostgresRagStore()
