# Production Readiness

This document records the verified release shape of the MCP-Enabled Weather Intelligence System.

## Verified application state

- Automated suite: **122 tests passing locally**.
- Agent E2E evaluation: **16/16 cases passing** in the verified release run.
- Tool-selection accuracy: **100%** in the verified agent evaluation.
- Argument accuracy: **100%** in the verified agent evaluation.
- Evidence sufficiency: **100%** in the verified agent evaluation.
- Agent success rate: **100%** in the verified agent evaluation.
- Unnecessary tool-call rate: **0%** in the verified agent evaluation.
- Unexpected tool-call rate: **0%** in the verified agent evaluation.
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

Live evaluations consume Gemini quota. The 16-case agent evaluation is the primary repeatable live correctness check; the larger stress suite should be run deliberately when quota and time permit.

## Release checklist

1. Rotate any credential that has ever been exposed.
2. Pull the latest `main` branch.
3. Run `python -m pytest -q`.
4. Build and start the Docker Compose service.
5. Verify `/healthz` and `/readyz`.
6. Run `python -m evaluation.agent_e2e_eval` with valid Gemini credentials.
7. Run the stress suite only when additional Gemini quota is available.
8. Verify the Railway deployment and inspect a representative trace.
9. Confirm `git status` is clean and no secrets are tracked.
