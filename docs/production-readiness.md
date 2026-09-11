# Production Readiness

This document records the verified release shape of the MCP-Enabled Weather Intelligence System.

## Verified application state

- Automated suite: **122 tests passing locally**.
- Balanced live agent stress suite: **100 generated evaluation cases** covering current weather, forecast, alerts, activity risk, and weather knowledge across multiple Indian cities.
- Railway deployment: live and verified through health/readiness checks and representative agent/RAG/API flows.

## Verified Docker state

- Production image builds successfully.
- API runs behind Gunicorn.
- Container runs with a non-root user.
- Container drops Linux capabilities.
- `no-new-privileges` is enabled.
- Root filesystem is read-only.
- `/tmp` is provided as a constrained tmpfs.
- `/healthz` provides liveness.
- `/readyz` checks required Gemini configuration and RAG-store initialization.
- Trace output is persisted through the dedicated writable observability volume.

## Runtime dependency boundary

The default API runtime intentionally excludes the optional dense-RAG ML stack (`torch` and `sentence-transformers`). Those dependencies live in `requirements-rag-ml.txt` and are needed only when dense retrieval is explicitly enabled.

## Configuration

Set `GEMINI_API_KEY` through the environment or a local `.env` file. Never commit API keys or generated evaluation/observability artifacts.

The default Docker stack uses PostgreSQL + pgvector for the RAG store. PostgreSQL full-text retrieval is the normal path; dense retrieval is opt-in.

## Evaluation note

Live evaluations consume Gemini quota. The balanced 100-case stress suite is the primary live agent evaluation and should be run deliberately when quota and time permit.

## Release checklist

1. Rotate any credential that has ever been exposed.
2. Pull the latest `main` branch.
3. Run `python -m pytest -q`.
4. Build and start the Docker Compose service.
5. Verify `/healthz` and `/readyz`.
6. Run the 100-case stress suite with valid Gemini credentials and sufficient quota.
7. Verify the Railway deployment and inspect a representative trace.
8. Confirm `git status` is clean and no secrets are tracked.
