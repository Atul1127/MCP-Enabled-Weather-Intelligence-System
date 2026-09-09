# Production AI Upgrade

This document summarizes the production-focused changes now present in `main`.

## 1. PostgreSQL + pgvector RAG

- PostgreSQL is the default RAG backend for the production Docker stack.
- `db/init.sql` creates the weather document and vector tables plus full-text and vector indexes.
- `rag/postgres_store.py` provides metadata-filtered PostgreSQL retrieval.
- PostgreSQL full-text retrieval is the default knowledge path.
- pgvector dense retrieval is available as an explicit option.
- The bundled JSONL corpus can be loaded idempotently into PostgreSQL.
- The local JSONL backend remains available for lightweight development/testing.

## 2. Bounded agent execution

The system retains the explainable LangGraph flow:

```text
Router -> Planner -> Decomposer -> Reasoner -> MCP Executor -> Verifier -> Synthesizer
```

Deterministic paths can avoid redundant model work when the required capability is already known. The RAG-only path can also bypass the full live-weather execution loop.

## 3. Layered security

The security boundary normalizes Unicode, removes common invisible characters before inspection, blocks control characters, checks prompt-injection signals, validates locations and queries, enforces MCP tool allowlists, bounds argument/observation size and nesting, and keeps synchronization disabled by default.

These controls are defense in depth and are not a complete prompt-injection guarantee.

## 4. Observability

Trace context propagates through the agent and MCP execution path. Nested spans record stage names, latency, success/failure state, retry attempts, and a shared trace ID.

Prompt content and secrets are not intentionally written to trace events.

## 5. Evaluation

The repository contains focused evaluation suites for retrieval, RAG, routing, answer quality, agent execution, and stress testing.

The verified release run includes:

- 122 automated tests passing locally.
- 16/16 live agent evaluation cases passing.
- 100% tool-selection accuracy in that evaluation.
- 100% argument accuracy in evaluable cases.
- 100% evidence sufficiency.

The larger stress evaluation is quota-consuming and should be run deliberately.

## Deployment

The application is deployed on Railway:

https://mcp-enabled-weather-intelligence-system-production.up.railway.app/

The local production-style deployment uses Docker Compose with PostgreSQL + pgvector and the same API surface.

## Runtime dependency boundary

The normal API image intentionally excludes `torch` and `sentence-transformers`. They remain isolated in `requirements-rag-ml.txt` and are needed only when dense retrieval is explicitly enabled.
