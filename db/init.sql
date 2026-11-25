CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS weather_documents (
    id TEXT PRIMARY KEY,
    location TEXT,
    state TEXT,
    district TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    source TEXT NOT NULL DEFAULT 'open-meteo',
    source_type TEXT NOT NULL,
    headline TEXT,
    narrative_text TEXT,
    forecast_date DATE,
    temperature_min_c DOUBLE PRECISION,
    temperature_max_c DOUBLE PRECISION,
    rainfall_mm DOUBLE PRECISION,
    precipitation_probability DOUBLE PRECISION,
    weather_code INTEGER,
    severity TEXT,
    issued_at TIMESTAMPTZ,
    payload JSONB NOT NULL,
    synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Generic reference documents may not have a physical weather location.
ALTER TABLE weather_documents ALTER COLUMN location DROP NOT NULL;

CREATE INDEX IF NOT EXISTS idx_weather_documents_location ON weather_documents(location);
CREATE INDEX IF NOT EXISTS idx_weather_documents_state ON weather_documents(state);
CREATE INDEX IF NOT EXISTS idx_weather_documents_source_type ON weather_documents(source_type);
CREATE INDEX IF NOT EXISTS idx_weather_documents_forecast_date ON weather_documents(forecast_date);
CREATE INDEX IF NOT EXISTS idx_weather_documents_issued_at ON weather_documents(issued_at);

CREATE TABLE IF NOT EXISTS weather_embeddings (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES weather_documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    model_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_weather_embeddings_document_id ON weather_embeddings(document_id);
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_embedding
    ON weather_embeddings USING hnsw (embedding vector_cosine_ops);

-- Full-text search column. It is generated from the text fields, so it is
-- always in sync, and the GIN index on it is used by "search_tsv @@ query".
-- The configuration ('english') must match FTS_CONFIG in rag/postgres_store.py.
ALTER TABLE weather_documents
    ADD COLUMN IF NOT EXISTS search_tsv tsvector
    GENERATED ALWAYS AS (
        to_tsvector(
            'english',
            COALESCE(location, '') || ' ' ||
            COALESCE(state, '') || ' ' ||
            COALESCE(district, '') || ' ' ||
            COALESCE(headline, '') || ' ' ||
            COALESCE(narrative_text, '')
        )
    ) STORED;

-- The old expression index never matched the query, so it was never used.
DROP INDEX IF EXISTS idx_weather_documents_fts;
CREATE INDEX IF NOT EXISTS idx_weather_documents_search_tsv
    ON weather_documents USING gin (search_tsv);
