"""PostgreSQL + pgvector retrieval backend."""
from __future__ import annotations

import os
import re
from typing import Any

import lakebase

# Must match idx_weather_documents_fts in db/init.sql and lakebase.ensure_weather_tables
# exactly, otherwise PostgreSQL cannot use the GIN index and scans the whole table.
_DOCUMENT_TSVECTOR = (
    "to_tsvector('simple', "
    "COALESCE(d.location, '') || ' ' || COALESCE(d.state, '') || ' ' || "
    "COALESCE(d.district, '') || ' ' || COALESCE(d.headline, '') || ' ' || "
    "COALESCE(d.narrative_text, ''))"
)

# Rows written by /weather/sync are point-in-time observations and forecasts.
# They go stale quickly and must not be served as weather *knowledge*.
_LIVE_SOURCE_TYPES = ("current", "forecast")

_DOCUMENT_COLUMNS = """
    d.id, d.location, d.state, d.district, d.source, d.source_type,
    d.headline, d.narrative_text, d.forecast_date, d.temperature_min_c,
    d.temperature_max_c, d.rainfall_mm, d.precipitation_probability,
    d.weather_code, d.severity
"""


class PostgresRagStore:
    """Database-backed RAG store using PostgreSQL full-text + pgvector retrieval."""

    def __init__(self) -> None:
        # Schema creation and corpus indexing belong to deployment/setup jobs,
        # never to an end-user retrieval request.
        self._require_schema = os.environ.get("WEATHER_RAG_REQUIRE_SCHEMA", "1").lower() not in {"0", "false", "no"}
        self._include_live_documents = os.environ.get("WEATHER_RAG_INCLUDE_LIVE_DOCS", "0").lower() in {"1", "true", "yes"}

    def check_ready(self) -> None:
        """Raise if the database (and, when required, the RAG schema) is unavailable."""
        if self._require_schema:
            if not lakebase.check_weather_schema():
                raise RuntimeError("RAG schema is missing; run scripts/init_database.py")
        elif not lakebase.check_connection():
            raise RuntimeError("Database connection check failed")

    @staticmethod
    def _escape_like(value: str) -> str:
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    @classmethod
    def _where(cls, location: str | None, state: str | None, source_type: str | None) -> tuple[str, list[Any]]:
        clauses = ["1 = 1"]
        params: list[Any] = []
        if location:
            clauses.append("d.location ILIKE %s")
            params.append(cls._escape_like(location))
        if state:
            clauses.append("d.state ILIKE %s")
            params.append(cls._escape_like(state))
        if source_type:
            clauses.append("d.source_type = %s")
            params.append(source_type)
        return " AND ".join(clauses), params

    def filtered_rows(self, location: str | None = None, state: str | None = None, source_type: str | None = None) -> list[str] | None:
        """Return allowed IDs for a filtered search, or None when nothing is filtered."""
        if not (location or state or source_type):
            return None
        where, params = self._where(location, state, source_type)
        return [str(row["id"]) for row in lakebase.run_query(f"SELECT d.id FROM weather_documents d WHERE {where}", tuple(params))]

    def _scope_clause(self, allowed: list[str] | None) -> tuple[str, list[Any]]:
        clause, params = "", []
        if not self._include_live_documents:
            clause += f" AND d.source_type NOT IN ({','.join(['%s'] * len(_LIVE_SOURCE_TYPES))})"
            params.extend(_LIVE_SOURCE_TYPES)
        if allowed is None:
            return clause, params
        if not allowed:
            return clause + " AND FALSE", params
        return clause + f" AND d.id IN ({','.join(['%s'] * len(allowed))})", params + list(allowed)

    @staticmethod
    def _fts_or_query(query: str) -> str:
        tokens = re.findall(r"[\w]+", query, flags=re.UNICODE)
        return " | ".join(token for token in tokens[:16] if len(token) >= 3)

    def _fts(self, tsquery_sql: str, query: str, limit: int, allowed: list[str] | None) -> list[dict[str, Any]]:
        extra, extra_params = self._scope_clause(allowed)
        rows = lakebase.run_query(f"""
            SELECT {_DOCUMENT_COLUMNS},
                   ts_rank_cd({_DOCUMENT_TSVECTOR}, {tsquery_sql}) AS bm25_score
            FROM weather_documents d
            WHERE {_DOCUMENT_TSVECTOR} @@ {tsquery_sql} {extra}
            ORDER BY bm25_score DESC, d.synced_at DESC NULLS LAST LIMIT %s
        """, tuple([query, query, *extra_params, int(limit)]))
        return [dict(row) for row in rows if float(row.get("bm25_score") or 0) > 0]

    def bm25_search(self, query: str, limit: int, allowed: list[str] | None = None) -> list[dict[str, Any]]:
        matches = self._fts("plainto_tsquery('simple', %s)", query, limit, allowed)
        if matches:
            return matches
        relaxed = self._fts_or_query(query)
        if not relaxed:
            return []
        return self._fts("to_tsquery('simple', %s)", relaxed, limit, allowed)

    def dense_search(self, query_vector: list[float], limit: int, allowed: list[str] | None = None) -> list[dict[str, Any]]:
        vector_literal = "[" + ",".join(repr(float(v)) for v in query_vector) + "]"
        extra, extra_params = self._scope_clause(allowed)
        rows = lakebase.run_query(f"""
            SELECT d.id, d.location, d.state, d.district, d.source, d.source_type,
                   d.headline, d.narrative_text, d.temperature_min_c, d.temperature_max_c,
                   d.rainfall_mm, d.precipitation_probability, d.weather_code, d.severity, e.chunk_text,
                   1 - (e.embedding <=> %s::vector) AS dense_score
            FROM weather_embeddings e JOIN weather_documents d ON d.id = e.document_id
            WHERE 1 = 1 {extra}
            ORDER BY e.embedding <=> %s::vector LIMIT %s
        """, tuple([vector_literal, *extra_params, vector_literal, int(limit)]))
        return [dict(row) for row in rows]


def get_store() -> PostgresRagStore:
    return PostgresRagStore()
