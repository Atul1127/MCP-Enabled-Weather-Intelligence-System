# Gemini Agent + Modular RAG

The main branch is a Gemini-only, MCP-first weather intelligence system with a modular local RAG stack.

## Architecture

```text
User
  |
  v
Gemini WeatherAgent
  |
  +--> Router / Planner
  |
  +--> MCP Executor
         |
         +--> live weather
         +--> forecast
         +--> hazard detection
         +--> activity risk
         +--> weather knowledge retrieval
                    |
                    v
               RAGPipeline
                    |
                    +--> query analysis
                    +--> query expansion
                    +--> metadata filtering
                    +--> dense retrieval
                    +--> BM25
                    +--> confidence-aware RRF
                    +--> cross-encoder reranking
                    +--> context compression
                    |
                    v
              Unified Evidence
                    |
                    v
             Gemini Synthesizer
                    |
                    v
             citation-aware answer
```

The MCP server is the capability boundary. The agent owns reasoning and orchestration. RAG owns retrieval and evidence preparation. The final Gemini synthesizer combines live MCP evidence with RAG evidence.

## Local RAG backend

The default RAG backend is file-backed and does not require PostgreSQL, Lakebase, Databricks credentials, or a paid vector database. The corpus is `data/weather_knowledge.jsonl`; dense embeddings use the local Sentence Transformers model and lexical retrieval uses BM25.

The PostgreSQL/Lakebase infrastructure remains available for managed weather-data ingestion and deployment paths.

## Gemini

Set:

```bash
export GEMINI_API_KEY="your-key"
export GEMINI_MODEL=gemini-3.6-flash
```

Optional fallback models and thinking settings are configured through the environment variables documented in the README.

## Evaluation

Run the repository tests first:

```bash
python -m pytest -q
```

Then run the retrieval and agent benchmarks:

```bash
python evaluation/retrieval_benchmark.py
python evaluation/rag_llm_eval.py
python evaluation/agent_benchmark.py
```

The evaluation stack measures retrieval quality, tool-selection accuracy, argument accuracy, latency, evidence relevance, faithfulness, and citation behavior.

## Observability

Set `WEATHER_TRACE_PATH` to change the JSONL destination. Agent, MCP, retrieval, reranking, compression, and synthesis stages emit trace events with a trace ID.
