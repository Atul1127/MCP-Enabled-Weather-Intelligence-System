# Deployment Guide

## Production deployment

The application is deployed on Railway:

**https://mcp-enabled-weather-intelligence-system-production.up.railway.app/**

Useful probes:

- `GET /healthz` — process liveness
- `GET /readyz` — Gemini + RAG readiness
- `GET /` — dashboard

Keep platform secrets and database credentials in the platform's environment configuration. Never commit them to the repository.

## Local Docker deployment

The repository's production-style local stack uses Flask/Gunicorn, PostgreSQL with pgvector, and the same application image used by the API service.

1. Copy `.env.example` to `.env`.
2. Set `GEMINI_API_KEY` to a valid key.
3. Start the stack:

```bash
docker compose up --build -d
```

4. Verify liveness and readiness:

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz
```

5. Open the dashboard at `http://localhost:8000/`.

The API container runs as a non-root user, drops Linux capabilities, enables `no-new-privileges`, uses a read-only root filesystem, and provides a constrained `/tmp` tmpfs. Trace output is stored on the dedicated writable observability volume.

## Configuration

Important environment variables include:

- `GEMINI_API_KEY`
- `GEMINI_MODEL`
- `GEMINI_FALLBACK_MODELS`
- `DATABASE_BACKEND` / PostgreSQL connection settings
- `WEATHER_RAG_BACKEND`
- `WEATHER_RAG_DENSE`
- `WEATHER_TRACE_PATH`
- `WEATHER_ALLOW_SYNC`

Dense retrieval is disabled by default. Enable it only when the optional ML requirements are installed.

Keep `WEATHER_ALLOW_SYNC=0` unless the synchronization endpoint is explicitly protected with an administrative credential and appropriate network policy.

## Production verification

Before declaring a deployment ready:

```bash
python -m pytest -q
WEATHER_STRESS_LIMIT=100 WEATHER_STRESS_REPORT=/tmp/stress_report_100.json python -m evaluation.stress_eval
```

Then verify the deployed `/healthz` and `/readyz` endpoints and inspect a representative trace when observability is enabled.

## Persistence

PostgreSQL is the default production RAG backend in the Docker configuration. The code also retains a local JSONL backend for development and a Lakebase-compatible persistence path for managed Databricks environments.

The bundled weather corpus can be indexed into PostgreSQL with:

```bash
python -m rag.index_local_corpus
```

## Secrets and artifacts

Never commit API keys, database passwords, trace logs, benchmark outputs containing sensitive data, or generated local state. Rotate any credential that has appeared in source control, terminal output, screenshots, logs, or chat.
