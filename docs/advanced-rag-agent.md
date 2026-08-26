# Advanced local agent + RAG

The upgrade branch turns the weather project into a local-first agentic RAG system.

## Architecture

```text
User
  |
  v
ReAct Agent (Ollama)
  |
  +--> MCP live weather tools
  |      +--> current forecast
  |      +--> alerts
  |      +--> activity risk
  |
  +--> MCP retrieval tools
         |
         +--> query expansion
         +--> metadata filtering
         +--> dense retrieval
         +--> BM25
         +--> Reciprocal Rank Fusion
         +--> cross-encoder reranking
         +--> context compression
         +--> grounded generation

All stages emit trace events to observability/traces.jsonl.
```

## Local backend

The default RAG backend is file-backed and does **not** require PostgreSQL,
Lakebase, Databricks credentials, an API key, or a paid service. The corpus is
`data/weather_knowledge.jsonl`; dense embeddings are built in memory with the
local Sentence Transformers model and BM25 runs locally.

The production PostgreSQL/Lakebase infrastructure remains in the repository for
managed deployments and for the existing weather-data ingestion path. It is not
required to run the local RAG demo.

Start Ollama and make sure the configured model is available (default:
`llama3.2:3b`). Then run:

```bash
python agent.py "What weather conditions are typically associated with heavy rainfall?"
```

## Advanced RAG

`advanced_rag.py` implements multi-query expansion, metadata filtering, dense
retrieval, BM25, RRF fusion, cross-encoder reranking, bounded context
compression, source citations, and structured trace IDs.

A failed retrieval returns an explicit grounded-retrieval failure instead of
falling back to the general LLM knowledge.

## Evaluation

Run:

```bash
python evaluation/evaluate_rag_local.py
python evaluation/evaluate_agent.py
```

The evaluators use the local Ollama model where an LLM judge is required. They
also report retrieval/agent metrics without requiring an external telemetry or
LLM API service.

## Observability

Set `WEATHER_TRACE_PATH` to change the JSONL destination. Each agent/RAG run
gets a trace ID and emits spans for reasoning, tool execution, retrieval, query
expansion, reranking, context construction, and generation. No external
telemetry service is required.
