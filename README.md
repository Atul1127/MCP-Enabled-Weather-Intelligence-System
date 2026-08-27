# MCP-Enabled Weather Intelligence System

An **MCP-first weather intelligence platform** for Indian locations. The system combines real-time weather APIs, deterministic hazard intelligence, activity-risk assessment, modular hybrid RAG, Gemini tool-calling, LangGraph orchestration, a Flask dashboard/API, observability, security gates, and evaluation.

## Architecture

```text
User / HTTP API
       |
       v
LangGraph WeatherAgent
       |
       +--> Router -> Planner -> Decomposer
       |
       +--> Reasoner <---- bounded recovery ---- Verifier
       |       |
       |       +--> MCP Executor
       |              |
       |              +--> current weather
       |              +--> forecast
       |              +--> hazard detection
       |              +--> activity risk
       |              +--> weather knowledge retrieval
       |                         |
       |                         v
       |                    RAGPipeline
       |                         |
       |                         +--> Query analysis / expansion
       |                         +--> Metadata filtering
       |                         +--> Dense retrieval
       |                         +--> BM25
       |                         +--> Confidence-aware RRF
       |                         +--> Cross-encoder reranking
       |                         +--> Context compression
       |                         |
       |                         v
       +-------------------- Unified Evidence
                              |
                              v
                       Gemini Synthesizer
                              |
                     Structured response
                              |
                     Citation validation
                              |
                              v
                           Answer
```

**Gemini is the only LLM provider. Ollama is not required.** All Gemini text, structured-output, and tool-calling requests pass through the shared `llm_provider.py` gateway, which provides retries and configured model fallback.

## Highlights

- **Gemini function calling** for agent routing and MCP tool selection
- **LangGraph** for explicit routing, planning, tool loops, verification, and bounded recovery
- **MCP v2** dynamic tool discovery over stdio
- **Live weather + 7-day forecast** through Open-Meteo
- **Deterministic hazard intelligence** for rain, wind, heat, and thunderstorms
- **Deterministic activity-risk scoring** from forecast signals
- **Hybrid RAG** using dense retrieval + BM25 + confidence-aware RRF
- **Cross-encoder reranking**
- **Query-aware context compression**
- **Typed evidence layer** separating live weather, risk, alerts, and RAG evidence
- **Structured Gemini synthesis** with schema validation and citation validation
- **Deterministic MCP input/output security boundaries**
- **Flask dashboard and separate RAG/agent HTTP endpoints**
- **Trace-level observability** for routing, MCP, RAG, reranking, and generation
- **Retrieval, RAG, agent, and answer-quality evaluation suites**

## Gemini configuration

Set the API key in the environment:

```bash
export GEMINI_API_KEY="your-key"
```

Optional configuration:

```bash
export GEMINI_MODEL=gemini-3.6-flash
export GEMINI_FALLBACK_MODELS=gemini-3.5-flash-lite,gemini-2.5-flash-lite
export GEMINI_THINKING_LEVEL=low
export GEMINI_MAX_OUTPUT_TOKENS=700
```

The same settings are used by RAG generation, agent tool selection, and final structured synthesis.

## Setup

### Git Bash on Windows

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Environment template

Copy `.env.example` to `.env` and fill in `GEMINI_API_KEY` and local database credentials when using Docker Compose. `.env` is ignored by Git.

### MCP smoke test

```bash
python mcp_client.py
```

### CLI agent

```bash
python agent.py "What weather conditions are typically associated with heavy rainfall?"
```

`weather_agent.py` is retained only as a backward-compatible wrapper around the same canonical `WeatherAgent` implementation.

## Flask

Start the dashboard/API:

```bash
python app.py
```

Useful endpoints:

```text
GET  /healthz
POST /weather/current      Direct live weather endpoint
POST /weather/alerts       Direct deterministic hazard endpoint
POST /weather/ask          RAG-only knowledge endpoint
POST /weather/agent        Full LangGraph + MCP + RAG agent
POST /weather/sync         Protected write endpoint (disabled by default)
```

The `/weather/ask` route intentionally remains a lightweight RAG endpoint. Use `/weather/agent` when you want the complete agentic workflow.

## Docker Compose

Create `.env` from `.env.example`, set `POSTGRES_PASSWORD` and `GEMINI_API_KEY`, then run:

```bash
docker compose up --build
```

Database and API credentials are supplied through environment variables rather than committed Docker defaults. Synchronization remains disabled unless explicitly enabled and authenticated.

## MCP tools

| Tool | Purpose |
|---|---|
| `get_weather` | Current weather and forecast data |
| `get_forecast` | Specific future-day forecast |
| `get_weather_alerts` | Forecast-based hazard detection |
| `assess_weather_risk` | Deterministic activity-risk assessment |
| `search_weather` | Hybrid weather knowledge retrieval and evidence preparation |
| `ask_weather` | Alias for weather knowledge evidence retrieval |
| `sync_weather` | Fetch and persist fresh weather data; not exposed to the model allowlist |
| `database_health` | Lakebase/PostgreSQL health check |

The model-facing allowlist deliberately excludes write-capable `sync_weather`. MCP discovery can still expose server capabilities, while the agent executor enforces the application policy.

## Modular RAG

```text
Query
  |
  +--> Query analysis
  +--> Query expansion when needed
  +--> Metadata filtering
  +--> Dense retrieval
  +--> BM25 retrieval
          |
          v
   Confidence-aware RRF
          |
          v
   Cross-encoder reranking
          |
          v
   Context compression
          |
          v
      RAG evidence
```

The local benchmark path uses the file-backed `LocalRagStore`. PostgreSQL/Lakebase remains available for managed weather-data persistence. The final agent answer is generated by Gemini outside the retrieval layer.

## Security

The agent applies deterministic prompt-injection signal checks to user input. MCP tool arguments are validated before execution, and untrusted MCP results are bounded by type, size, and nesting depth. Both the reasoning and final synthesis prompts explicitly treat tool output and retrieved content as untrusted data. These are defense-in-depth controls, not a complete prompt-injection solution.

## Observability

Agent, MCP, retrieval, reranking, context compression, and synthesis stages emit trace events with a trace ID. The MCP stdio environment propagates `WEATHER_TRACE_ID`, so server-side MCP events can be correlated with the originating agent run. Set `WEATHER_TRACE_PATH` to change the JSONL destination.

Inspect a trace with:

```bash
python evaluation/trace_report.py <trace_id>
```

## Evaluation

Run the full unit/integration suite:

```bash
python -m pytest -q
```

For quota-friendly live agent checks, run only a small subset first:

```bash
python -m evaluation.agent_e2e_smoke --limit 1
python -m evaluation.agent_e2e_smoke --limit 3 --category live_weather
```

Full live benchmark:

```bash
python evaluation/agent_e2e_eval.py
```

Other evaluation suites:

```bash
python evaluation/retrieval_benchmark.py
python evaluation/rag_llm_eval.py
python evaluation/agent_benchmark.py
```

The evaluation stack measures retrieval quality, tool-selection accuracy, argument accuracy, latency, evidence sufficiency, and citation behavior. Retrieval/LLM evaluation separately covers answer-quality dimensions where applicable.

## Key technologies

- **Gemini API / Google GenAI SDK** — agent reasoning, tool calling, and grounded synthesis
- **LangGraph** — explicit agent orchestration and bounded recovery
- **MCP v2** — protocol-based tool discovery and capability boundary
- **Open-Meteo** — weather and forecast data
- **Sentence Transformers** — dense embeddings and cross-encoder reranking
- **BM25** — lexical retrieval
- **RRF** — hybrid ranking
- **Flask** — API/dashboard
- **PostgreSQL / pgvector** — persistence/vector search where configured
- **Python 3.13** — application, MCP implementation, CI, and container runtime

## Project structure

```text
agent.py                    Canonical CLI + compatibility API
weather_agent_core/         Router, planner, LangGraph, executor, evidence, synthesis
llm_provider.py             Single Gemini generation/tool-calling gateway
mcp_client.py               MCP stdio client + capability discovery
mcp_server.py               MCP server + weather/RAG tools
rag/                        Modular RAG pipeline
rag_service.py              Thin HTTP-facing RAG adapter
app.py                      Flask dashboard/API
evaluation/                 Benchmarks, traces, and answer evaluation
tests/                      Unit and integration tests
docs/                       Architecture/documentation
.env.example                Deployment configuration template
```

## Design goal

> **Gemini decides which capability is needed, LangGraph controls the execution loop, MCP provides the capability boundary, deterministic weather logic handles safety-sensitive scoring, and modular hybrid RAG provides grounded domain knowledge.**

The evidence layer keeps live observations, forecasts, risk classifications, alerts, and retrieved knowledge distinct so the final answer can remain grounded and auditable.
