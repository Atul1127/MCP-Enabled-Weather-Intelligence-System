# Deployment Guide

## Default Docker deployment

The repository's default Compose deployment uses the file-backed local RAG store and is intended for a simple single-container deployment.

1. Copy `.env.example` to `.env`.
2. Set `GEMINI_API_KEY` to a valid key.
3. Build and start the service:

```bash
docker compose up --build -d
```

4. Verify liveness and readiness:

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz
```

The container runs as a non-root user, drops Linux capabilities, enables `no-new-privileges`, uses a read-only root filesystem, and provides a constrained `/tmp` tmpfs.

## Hosted deployment checklist

For a hosted environment, provide the following as platform-managed configuration/secrets rather than committing them:

- `GEMINI_API_KEY`
- `GEMINI_MODEL`
- `GEMINI_FALLBACK_MODELS`
- `WEATHER_TRACE_PATH`
- Persistence/database settings when using PostgreSQL/Lakebase instead of the local store.

Expose only port `8000` behind the platform's TLS/reverse proxy. Configure the platform health check to use `/readyz` and retain `/healthz` for process liveness.

Set a restart policy and resource limits appropriate to the platform. Keep `WEATHER_ALLOW_SYNC=0` unless the write-capable synchronization endpoint has been explicitly secured with an administrative credential and network policy.

## Production verification

Before declaring a deployment ready:

```bash
python -m pytest -q
python -m evaluation.agent_e2e_eval
python -m evaluation.agent_benchmark
```

Then verify the deployed `/healthz` and `/readyz` endpoints and inspect a representative trace if observability is enabled.

## Persistence options

The default local deployment is intentionally simple and zero-cost. PostgreSQL/Lakebase support remains available for environments that need managed persistence. Select that backend explicitly and provide its credentials through the deployment platform's secret/configuration system.

## Secrets and artifacts

Never commit API keys, database passwords, trace logs, benchmark outputs containing sensitive data, or generated local state. Rotate any credential that has appeared in source control, terminal output, screenshots, logs, or chat.
