"""PostgreSQL + pgvector retrieval backend."""
from __future__ import annotations

from typing import Any

import lakebase


class PostgresRagStore:
    """Database-backed RAG store using PostgreSQL full-text + pgvector retrieval."""

    def __init__(self) -> None:
        lakebase.ensure_weather_tables(embedding_dim=384)
        self._bootstrap_corpus()

    @staticmethod
    def _bootstrap_corpus() -> None:
        rows = lakebase.run_query("SELECT COUNT(*) AS count FROM weather_documents WHERE source = 'local-weather-guide'")
        if rows and int(rows[0]["count"]) > 0:
            return
        from rag.index_local_corpus import load_corpus
        load_corpus()

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
        return [str(row["id"]) for row in lakebase.run_query(f"SELECT d.id FROM weather_documents d WHERE {where}", tuple(params))]

    def _ids_clause(self, allowed: list[str] | None) -> tuple[str, list[Any]]:
        # None means no filtering was requested. An empty list means filtering
        # was requested but no documents matched; never silently fall back to
        # the entire corpus in that case.
        if allowed is None:
            return "", []
        if not allowed:
            return " AND FALSE", []
        return f" AND d.id IN ({','.join(['%s'] * len(allowed))})", list(allowed)

    @staticmethod
    def _fts_or_query(query: str) -> str:
        """Build a safe OR tsquery from natural-language tokens.

        plainto_tsquery uses AND semantics, which can make a normal question
        require every word to occur in one document. Tokenizing here and joining
        normalized terms with OR keeps lexical retrieval useful for long queries.
        """
        import re
        tokens = re.findall(r"[\w]+", query, flags=re.UNICODE)
        tokens = [token for token in tokens if len(token) >= 3]
        return " | ".join(tokens[:16])

    def bm25_search(self, query: str, limit: int, allowed: list[str] | None = None) -> list[dict[str, Any]]:
        extra, extra_params = self._ids_clause(allowed)
        rows = lakebase.run_query(f"""
            SELECT d.id, d.location, d.state, d.district, d.source, d.source_type,
                   d.headline, d.narrative_text, d.forecast_date, d.temperature_min_c,
                   d.temperature_max_c, d.rainfall_mm, d.precipitation_probability,
                   d.weather_code, d.severity,
                   ts_rank_cd(to_tsvector('simple', concat_ws(' ', d.location, d.state, d.district, d.headline, d.narrative_text)), plainto_tsquery('simple', %s)) AS bm25_score
            FROM weather_documents d WHERE 1 = 1 {extra}
            ORDER BY bm25_score DESC, d.synced_at DESC NULLS LAST LIMIT %s
        """, tuple([query, *extra_params, int(limit)]))
        matches = [dict(row) for row in rows if float(row.get("bm25_score") or 0) > 0]
        if matches:
            return matches

        relaxed = PostgresRagStore._fts_or_query(query)
        if not relaxed:
            return []
        rows = lakebase.run_query(f"""
            SELECT d.id, d.location, d.state, d.district, d.source, d.source_type,
                   d.headline, d.narrative_text, d.forecast_date, d.temperature_min_c,
                   d.temperature_max_c, d.rainfall_mm, d.precipitation_probability,
                   d.weather_code, d.severity,
                   ts_rank_cd(to_tsvector('simple', concat_ws(' ', d.location, d.state, d.district, d.headline, d.narrative_text)), to_tsquery('simple', %s)) AS bm25_score
            FROM weather_documents d WHERE 1 = 1 {extra}
            ORDER BY bm25_score DESC, d.synced_at DESC NULLS LAST LIMIT %s
        """, tuple([relaxed, *extra_params, int(limit)]))
        return [dict(row) for row in rows if float(row.get("bm25_score") or 0) > 0]

    def dense_search(self, query_vector: list[float], limit: int, allowed: list[str] | None = None) -> list[dict[str, Any]]:
        vector_literal = "[" + ",".join(repr(float(v)) for v in query_vector) + "]"
        extra, extra_params = self._ids_clause(allowed)
        rows = lakebase.run_query(f"""
            SELECT d.id, d.location, d.state, d.district, d.source, d.source_type,
                   d.headline, d.narrative_text, d.forecast_date, d.temperature_min_c,
                   d.temperature_max_c, d.rainfall_mm, d.precipitation_probability,
                   d.weather_code, d.severity, e.chunk_text,
                   1 - (e.embedding <=> %s::vector) AS dense_score
            FROM weather_embeddings e JOIN weather_documents d ON d.id = e.document_id
            WHERE 1 = 1 {extra}
            ORDER BY e.embedding <=> %s::vector LIMIT %s
        """, tuple([vector_literal, *extra_params, vector_literal, int(limit)]))
        return [dict(row) for row in rows]


def get_store() -> PostgresRagStore:
    return PostgresRagStore()
