# Production Readiness

This document records the verified release state of the MCP-Enabled Weather Intelligence System.

## Verified application state

- Unit/integration suite: 115 tests passing locally.
- Agent E2E evaluation: 16/16 cases passing.
- Tool-selection accuracy: 100%.
- Argument accuracy: 100%.
- Evidence sufficiency: 100%.
- Agent success rate: 100%.
- Unnecessary tool-call rate: 0%.
- Unexpected tool-call rate: 0%.
- Infrastructure failures: 0%.

## Verified Docker state

- Production image builds successfully.
- API runs behind Gunicorn.
- Container runs with a non-root user.
- Container drops all Linux capabilities.
- `no-new-privileges` is enabled.
- Root filesystem is read-only.
- `/tmp` is provided as a constrained tmpfs.
- `/healthz` returns HTTP 200 when the process is alive.
- `/readyz` returns HTTP 200 only when the Gemini configuration and RAG store are usable.

## Runtime dependency boundary

The default API image intentionally excludes the optional dense-RAG ML stack (`torch` and `sentence-transformers`). Those dependencies live in the separate ML requirements file and are loaded only when dense retrieval or ML reranking is used.

This keeps the default container substantially smaller and avoids pulling the CUDA/NVIDIA dependency tree during normal API builds.

## Configuration

Set `GEMINI_API_KEY` through the environment or a local `.env` file. Never commit API keys or generated evaluation/observability artifacts.

The local Docker Compose deployment uses the file-backed RAG store (`DATABASE_BACKEND=local`). PostgreSQL/Lakebase support remains available for deployments that explicitly configure the managed persistence path.

## Final performance reference

The latest 16-case benchmark recorded:

- Mean latency: approximately 8.6 seconds.
- P50 latency: approximately 8.6 seconds.
- P95 latency: approximately 12.4 seconds.

These are reference measurements rather than hard SLOs. Correctness and zero-error behavior are currently the release gate.

## Release checklist

1. Rotate any credential that has ever been exposed in terminal output, screenshots, logs, or chat.
2. Pull the latest `main` branch.
3. Run `pytest -q`.
4. Build and start the Docker Compose service.
5. Verify `/healthz` and `/readyz`.
6. Run `python -m evaluation.agent_e2e_eval` with valid Gemini credentials.
7. Run `python -m evaluation.agent_benchmark` and archive the generated report outside Git if it is not intended as a repository artifact.
8. Confirm `git status` is clean and no secrets are tracked.
