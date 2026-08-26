# MCP-Enabled Weather Intelligence System

An **MCP-first weather intelligence platform** for Indian locations. The system combines real-time weather APIs, deterministic hazard intelligence, activity-risk assessment, hybrid RAG, Gemini tool-calling, a Flask dashboard/API, Chainlit chat, observability, and evaluation.

## Architecture

```text
User
  |
  +--> Chainlit Chat / Flask API
  |
  v
Gemini Agent (tool calling)
  |
  v
MCP Client
  |
  v
MCP Server
  +--> get_weather
  +--> get_forecast
  +--> get_weather_alerts
  +--> assess_weather_risk
  +--> search_weather
  |
  v
Hybrid RAG
  +--> Dense retrieval
  +--> BM25
  +--> RRF / confidence-aware fusion
  +--> Cross-encoder reranking
  +--> Context compression
  +--> Gemini grounded generation
```

**Gemini is the only LLM provider in the application. Ollama is not required.** MCP remains the integration boundary between the model and weather capabilities.

## Highlights

- **Gemini function calling** for agent routing and MCP tool selection
- **MCP v2** dynamic tool discovery over stdio
- **Live weather + 7-day forecast** through Open-Meteo
- **Deterministic hazard intelligence** for rain, wind, heat, and thunderstorms
- **Deterministic activity-risk scoring** from forecast signals
- **Hybrid RAG** using dense retrieval + BM25 + RRF
- **Cross-encoder reranking** with warm/cold latency instrumentation
- **Grounded Gemini synthesis** with citation preservation
- **Chainlit conversational UI**
- **Flask dashboard and HTTP API**
- **Trace-level observability** for routing, MCP, RAG, reranking, and generation
- **Retrieval, RAG, agent, and answer-quality evaluation suites**

## Gemini configuration

Set the API key in the environment:

```bash
export GEMINI_API_KEY="your-key"
```

Optional model configuration:

```bash
export GEMINI_MODEL=gemini-3.6-flash
export GEMINI_FALLBACK_MODELS=gemini-3.5-flash-lite,gemini-2.5-flash-lite
export GEMINI_THINKING_LEVEL=low
export GEMINI_MAX_OUTPUT_TOKENS=700
```

The application automatically retries transient Gemini 429/5xx capacity errors and can fall back to the configured Gemini models.

## Setup

### Git Bash on Windows

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### MCP smoke test

```bash
python mcp_client.py
```

### CLI agent

```bash
python agent.py "What weather conditions are typically associated with heavy rainfall?"
```

The legacy `weather_agent.py` command is now a compatibility wrapper around the same Gemini agent:

```bash
python weather_agent.py "What is the weather in Kolkata right now?"
```

## Chainlit

Start the conversational UI:

```bash
chainlit run chainlit_app.py -w
```

The UI displays the active Gemini model and uses the same canonical MCP agent as the CLI.

## Flask

Start the dashboard/API:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:8000/
```

Useful endpoints include:

```text
GET  /healthz
POST /weather/current
POST /weather/alerts
POST /weather/ask
POST /weather/sync
```

## MCP tools

| Tool | Purpose |
|---|---|
| `get_weather` | Current weather and forecast data |
| `get_forecast` | Specific future-day forecast |
| `get_weather_alerts` | Forecast-based hazard detection |
| `assess_weather_risk` | Deterministic activity-risk assessment |
| `search_weather` | Hybrid weather knowledge retrieval + grounded answer |
| `ask_weather` | Existing grounded RAG answer path |

## Hybrid RAG

```text
Query
  |
  +--> Dense retrieval
  +--> BM25 retrieval
          |
          v
     RRF / confidence-aware fusion
          |
          v
     Cross-encoder reranking
          |
          v
     Context compression
          |
          v
     Gemini generation
          |
          v
     Citation processing
```

The retrieval layer is local and does not require an external vector database for the local benchmark path. Gemini is used for query/generation stages where configured by the RAG implementation.

## Observability

Every agent request can produce a trace covering stages such as:

```text
agent.reason
agent.route
agent.execute_tools
mcp.search_weather
rag.store_load
rag.query_planning
retrieval
reranking
context_compression
prompt_construction
generation
citation_processing
```

Inspect a trace with:

```bash
python evaluation/trace_report.py <trace_id>
```

## Evaluation

Run the full unit/integration suite:

```bash
python -m pytest -q
```

Retrieval benchmark:

```bash
python evaluation/retrieval_benchmark.py
```

Local RAG/LLM evaluation:

```bash
python evaluation/rag_llm_eval.py
```

Agent benchmark:

```bash
python evaluation/agent_benchmark.py
```

The project evaluates both implementation correctness and model behavior. Retrieval metrics include Hit@1, Recall@5, MRR, and latency percentiles; RAG evaluation includes success, evidence recall, faithfulness, relevance, evidence relevance, and citation validity.

## Key technologies

- **Gemini API / Google GenAI SDK** — agent reasoning, tool calling, and grounded generation
- **MCP v2** — protocol-based tool discovery and orchestration
- **Open-Meteo** — weather and forecast data
- **Sentence Transformers** — dense embeddings and reranking
- **BM25** — lexical retrieval
- **RRF** — hybrid ranking
- **Flask** — API/dashboard
- **Chainlit** — conversational agent UI
- **PostgreSQL / pgvector** — persistence/vector search where configured
- **Python** — application and MCP implementation

## Project structure

```text
agent.py                    Canonical Gemini MCP agent
weather_agent.py            Backward-compatible CLI wrapper
llm_provider.py             Gemini-only generation provider
mcp_client.py               MCP stdio client + tool discovery
mcp_server.py               MCP server + weather/RAG tools
rag_service.py              Hybrid retrieval/RAG pipeline
advanced_rag.py             Advanced retrieval components
chainlit_app.py             Chainlit conversational UI
app.py                      Flask dashboard/API
evaluation/                 Benchmarks, traces, and answer evaluation
tests/                      Unit and integration tests
docs/                       Architecture/documentation
```

## Design goal

The project demonstrates a production-oriented pattern:

> **Gemini decides which capability is needed, MCP provides the capability boundary, deterministic weather logic handles safety-sensitive scoring, and hybrid RAG provides grounded domain knowledge.**

This keeps model reasoning, tools, retrieval, and application logic cleanly separated and observable.
