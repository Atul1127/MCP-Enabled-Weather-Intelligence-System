"""Database connection helper for local PostgreSQL and Databricks Lakebase."""
from __future__ import annotations

import base64
import os
from contextlib import contextmanager
from urllib.parse import urlsplit, urlunsplit

import psycopg
from psycopg.rows import dict_row
from sqlalchemy import create_engine

DATABASE_BACKEND = os.environ.get("DATABASE_BACKEND", "local").lower().strip()
LOCAL_DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://weather_user:weather_password@localhost:5432/weather_rag")
LAKEBASE_SECRET_SCOPE = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
LAKEBASE_SECRET_KEY = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


DB_CONNECT_TIMEOUT_SECONDS = _int_env("WEATHER_DB_CONNECT_TIMEOUT", 10)
DB_STATEMENT_TIMEOUT_MS = _int_env("WEATHER_DB_STATEMENT_TIMEOUT_MS", 10000)
DB_LOCK_TIMEOUT_MS = _int_env("WEATHER_DB_LOCK_TIMEOUT_MS", 5000)
_workspace_client = None


def _get_workspace_client():
    global _workspace_client
    if _workspace_client is None:
        from databricks.sdk import WorkspaceClient
        _workspace_client = WorkspaceClient()
    return _workspace_client


def _get_lakebase_url() -> str:
    secret = _get_workspace_client().secrets.get_secret(scope=LAKEBASE_SECRET_SCOPE, key=LAKEBASE_SECRET_KEY)
    return base64.b64decode(secret.value).decode("utf-8")


def get_database_url() -> str:
    if DATABASE_BACKEND == "local":
        return LOCAL_DATABASE_URL
    if DATABASE_BACKEND == "lakebase":
        return _get_lakebase_url()
    raise ValueError(f"Unsupported DATABASE_BACKEND: {DATABASE_BACKEND}. Use 'local' or 'lakebase'.")


def _masked_database_url(url: str) -> str:
    try:
        parsed = urlsplit(url)
        if parsed.password is None:
            return url
        username = parsed.username or ""
        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, f"{username}:***@{host}", parsed.path, parsed.query, parsed.fragment))
    except Exception:
        return "configured"


@contextmanager
def get_connection():
    """Yield a bounded PostgreSQL connection using dict rows."""
    connection = psycopg.connect(
        get_database_url(),
        row_factory=dict_row,
        connect_timeout=DB_CONNECT_TIMEOUT_SECONDS,
        options=f"-c statement_timeout={DB_STATEMENT_TIMEOUT_MS} -c lock_timeout={DB_LOCK_TIMEOUT_MS}",
    )
    try:
        yield connection
    finally:
        connection.close()


def get_engine():
    database_url = get_database_url()
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    elif database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql+psycopg://", 1)
    return create_engine(
        database_url,
        connect_args={
            "connect_timeout": DB_CONNECT_TIMEOUT_SECONDS,
            "options": f"-c statement_timeout={DB_STATEMENT_TIMEOUT_MS} -c lock_timeout={DB_LOCK_TIMEOUT_MS}",
        },
        pool_pre_ping=True,
    )


def run_query(sql: str, params: tuple | dict | None = None) -> list[dict]:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchall()


def run_write(sql: str, params: tuple | dict | None = None) -> int:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            connection.commit()
            return cursor.rowcount


def check_connection() -> bool:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            return bool(cursor.fetchone())


def check_weather_schema() -> bool:
    """Verify RAG tables exist without mutating the database."""
    rows = run_query("SELECT to_regclass('public.weather_documents') AS documents_table, to_regclass('public.weather_embeddings') AS embeddings_table")
    return bool(rows and rows[0].get("documents_table") and rows[0].get("embeddings_table"))


def ensure_weather_tables(embedding_dim: int = 384) -> None:
    """Explicitly initialize the weather RAG schema; not part of retrieval."""
    run_write("CREATE EXTENSION IF NOT EXISTS vector;")
    run_write("""
        CREATE TABLE IF NOT EXISTS weather_documents (
            id TEXT PRIMARY KEY, location TEXT, state TEXT, district TEXT,
            latitude DOUBLE PRECISION, longitude DOUBLE PRECISION,
            source TEXT NOT NULL DEFAULT 'open-meteo', source_type TEXT NOT NULL,
            headline TEXT, narrative_text TEXT, forecast_date DATE,
            temperature_min_c DOUBLE PRECISION, temperature_max_c DOUBLE PRECISION,
            rainfall_mm DOUBLE PRECISION, precipitation_probability DOUBLE PRECISION,
            weather_code INTEGER, severity TEXT, issued_at TIMESTAMPTZ,
            payload JSONB NOT NULL, synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    migrations = [
        ("state", "TEXT"), ("district", "TEXT"), ("latitude", "DOUBLE PRECISION"),
        ("longitude", "DOUBLE PRECISION"), ("source", "TEXT DEFAULT 'open-meteo'"),
        ("forecast_date", "DATE"), ("temperature_min_c", "DOUBLE PRECISION"),
        ("temperature_max_c", "DOUBLE PRECISION"), ("rainfall_mm", "DOUBLE PRECISION"),
        ("precipitation_probability", "DOUBLE PRECISION"), ("weather_code", "INTEGER"), ("severity", "TEXT"),
    ]
    for column_name, column_type in migrations:
        run_write(f"ALTER TABLE weather_documents ADD COLUMN IF NOT EXISTS {column_name} {column_type}")
    indexes = [
        ("idx_weather_documents_location", "location"), ("idx_weather_documents_state", "state"),
        ("idx_weather_documents_district", "district"), ("idx_weather_documents_source", "source"),
        ("idx_weather_documents_source_type", "source_type"), ("idx_weather_documents_forecast_date", "forecast_date"),
        ("idx_weather_documents_precipitation_probability", "precipitation_probability"), ("idx_weather_documents_issued_at", "issued_at"),
    ]
    for index_name, column_name in indexes:
        run_write(f"CREATE INDEX IF NOT EXISTS {index_name} ON weather_documents ({column_name})")
    run_write("""
        CREATE INDEX IF NOT EXISTS idx_weather_documents_fts ON weather_documents USING gin (
            to_tsvector('simple', concat_ws(' ', location, state, district, headline, narrative_text))
        )
    """)
    run_write(f"""
        CREATE TABLE IF NOT EXISTS weather_embeddings (
            id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES weather_documents(id) ON DELETE CASCADE,
            chunk_index INT NOT NULL, chunk_text TEXT NOT NULL, embedding VECTOR({int(embedding_dim)}) NOT NULL,
            model_name TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), UNIQUE(document_id, chunk_index)
        )
    """)
    run_write("CREATE INDEX IF NOT EXISTS idx_weather_embeddings_document_id ON weather_embeddings (document_id)")
    run_write("CREATE INDEX IF NOT EXISTS idx_weather_embeddings_embedding ON weather_embeddings USING hnsw (embedding vector_cosine_ops)")


if __name__ == "__main__":
    print("Database backend:", DATABASE_BACKEND)
    print("Database URL:", _masked_database_url(get_database_url()))
    try:
        print("Database connection:", "OK" if check_connection() else "FAILED")
        print("Weather schema:", "OK" if check_weather_schema() else "MISSING")
    except Exception as exc:
        print("Database connection: FAILED")
        print(f"{type(exc).__name__}: {exc}")
