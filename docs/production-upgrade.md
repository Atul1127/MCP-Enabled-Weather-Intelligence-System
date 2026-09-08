# Production AI Upgrade

This branch upgrades the Weather Intelligence System around five high-impact goals.

## 1. PostgreSQL + pgvector

- PostgreSQL is now the default RAG backend for the production compose stack.
- `db/init.sql` creates `weather_documents`, `weather_embeddings`, HNSW vector indexing, and a PostgreSQL full-text index.
- `rag/postgres_store.py` performs metadata-filtered lexical retrieval and pgvector similarity retrieval.
- The bundled JSONL corpus is loaded idempotently into PostgreSQL.
- The original local JSONL store remains available with `WEATHER_RAG_BACKEND=local`.
- Dense retrieval is opt-in with `WEATHER_RAG_DENSE=1` because the ML dependencies are intentionally outside the lightweight runtime image.

## 2. Adaptive agent execution

The existing Router -> Planner -> Reasoner -> MCP -> Verifier -> Synthesis design is retained because it is useful for multi-step questions. Deterministic plans can bypass a redundant Gemini tool-selection round; the existing RAG-only direct path remains in place. This keeps the architecture explainable while avoiding model calls when the capability is already known.

## 3. Layered security

The security boundary now normalizes Unicode with NFKC, removes common zero-width/invisible characters before inspection, blocks control characters, retains injection signal checks, and continues to enforce tool allowlists, argument size/depth limits, semantic validation, and bounded untrusted observations.

These controls are defense in depth and are not a complete prompt-injection guarantee.

## 4. Observability

Trace context now propagates through async spans. MCP calls and final synthesis emit nested spans containing latency, success/failure state and retry attempts. Trace summaries also report failed spans.

No prompt or secret is written into the trace events by these changes.

## 5. Evaluation

`evaluation/stress_eval.py` generates a deterministic 512-case suite from multiple cities and paraphrased intent families. It evaluates the real Gemini + MCP loop rather than a mocked planner and reports task success, tool selection, argument correctness, mean latency, P50 and P95.

Use a small limit first:

```bash
WEATHER_STRESS_LIMIT=32 python -m evaluation.stress_eval
```

Then run the full suite when quota/time permits:

```bash
python -m evaluation.stress_eval
```

## Important deployment note

The compose stack uses PostgreSQL + pgvector and defaults to sparse retrieval because the normal API image intentionally excludes `sentence-transformers` and PyTorch. For a dense production image, install the ML requirements and set `WEATHER_RAG_DENSE=1`; the vectors are persisted in PostgreSQL rather than held as a full corpus in every API worker.
