# MCP-Enabled Weather Intelligence System

An **MCP-first weather intelligence platform for Indian locations** that combines live weather data, deterministic risk intelligence, hybrid RAG, Gemini tool calling, LangGraph orchestration, security boundaries, observability, and evaluation.

## What this project demonstrates

- **Agentic orchestration:** Gemini selects capabilities while LangGraph controls a bounded execution loop.
- **MCP architecture:** Weather and retrieval capabilities are exposed through protocol-based tools with an explicit model-facing allowlist.
- **Grounded RAG:** Dense retrieval + BM25 + confidence-aware RRF + reranking + context compression.
- **Safety-oriented intelligence:** Deterministic hazard detection and activity-risk scoring are separated from LLM generation.
- **Evidence-first answers:** Live weather, forecasts, risks, alerts, and retrieved knowledge remain typed and auditable before synthesis.
- **Production controls:** Input checks, MCP argument validation, bounded tool results, read-only containers, dropped capabilities, and no-new-privileges.
- **Evaluation:** Retrieval, RAG, agent, answer-quality, and end-to-end benchmark suites.

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
       |                    Hybrid RAG
       |                         |
       |                         +--> query analysis / expansion
       |                         +--> metadata filtering
       |                         +--> dense retrieval
       |                         +--> BM25
       |                         +--> confidence-aware RRF
       |                         +--> reranking
       |                         +--> context compression
       |
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

## Gemini

Gemini is the only LLM provider. All Gemini generation, structured-output, and tool-calling requests pass through the shared `llm_provider.py` gateway.

Set credentials through the environment or `.env`:

```bash
GEMINI_API_KEY=your-key
GEMINI_MODEL=gemini-3.6-flash
GEMINI_FALLBACK_MODELS=gemini-3.5-flash-lite
GEMINI_THINKING_LEVEL=low
GEMINI_MAX_OUTPUT_TOKENS=700
```

Never commit or paste real API keys. Rotate a key immediately if it has been exposed.

## Quick start

### Local Python

```bash
python -m venv .venv
source .venv/Scripts/activate       # Git Bash on Windows
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create `.env` from `.env.example`, set a valid Gemini key, then run:

```bash
python app.py
```

### Docker Compose

```bash
docker compose up --build -d
```

Verify the service:

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz
```

The default deployment uses the file-backed local RAG store. PostgreSQL/Lakebase support remains available when explicitly configured.

## Demo

Full reproducible examples are in [`docs/demo.md`](docs/demo.md).

Agent CLI:

```bash
python agent.py "What is the current weather in Mumbai and is it risky for outdoor activity?"
```

RAG endpoint:

```bash
curl -X POST http://localhost:8000/weather/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What conditions are associated with heavy rainfall?"}'
```

Full agent endpoint:

```bash
curl -X POST http://localhost:8000/weather/agent \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the current weather in Mumbai and what outdoor risks should I consider?"}'
```

## MCP tools

| Tool | Purpose |
|---|---|
| `get_weather` | Current weather and forecast data |
| `get_forecast` | Specific future-day forecast |
| `get_weather_alerts` | Deterministic forecast-based hazard detection |
| `assess_weather_risk` | Deterministic activity-risk assessment |
| `search_weather` | Hybrid weather knowledge retrieval |
| `ask_weather` | Weather knowledge evidence retrieval alias |
| `sync_weather` | Persistence sync; excluded from the model-facing allowlist |
| `database_health` | Persistence health check |

## Evaluation

Run unit/integration tests:

```bash
python -m pytest -q
```

Run the 16-case live agent evaluation:

```bash
python -m evaluation.agent_e2e_eval
```

Run the benchmark:

```bash
python -m evaluation.agent_benchmark
```

The latest verified 16-case benchmark achieved **100% task success, 100% tool-selection accuracy, 100% argument accuracy, 100% evidence sufficiency, 0% unnecessary calls, 0% unexpected calls, and 0% infrastructure failures**. Reference latency was approximately **8.6s mean / 8.6s P50 / 12.4s P95**.

## Runtime dependency boundary

The default API image keeps optional dense-RAG ML dependencies (`torch` and `sentence-transformers`) out of the runtime installation. They are listed in `requirements-rag-ml.txt` and loaded lazily only when dense retrieval or ML reranking is required.

This avoids pulling the large CUDA/NVIDIA dependency tree into the normal API image.

## Observability

Agent, MCP, retrieval, reranking, context-compression, and synthesis stages emit trace events with a shared trace ID. Inspect a trace with:

```bash
python evaluation/trace_report.py <trace_id>
```

Set `WEATHER_TRACE_PATH` to change the JSONL destination.

## Security

The system uses deterministic prompt-injection signal checks, MCP argument validation, bounded untrusted tool results, and prompts that treat tool output and retrieved content as untrusted data. Docker additionally uses a non-root runtime, dropped Linux capabilities, `no-new-privileges`, a read-only root filesystem, and a constrained `/tmp` tmpfs.

These controls are defense in depth, not a guarantee against every prompt-injection technique.

## Project structure

```text
agent.py                    Canonical CLI + compatibility API
weather_agent_core/         Router, planner, LangGraph, executor, evidence, synthesis
llm_provider.py             Shared Gemini gateway
mcp_client.py               MCP stdio client + capability discovery
mcp_server.py               MCP server + weather/RAG tools
rag/                        Modular RAG pipeline
rag_service.py              HTTP-facing RAG adapter
local_rag_store.py          File-backed local retrieval store
app.py                      Flask dashboard/API
evaluation/                 Benchmarks, traces, and answer evaluation
tests/                      Unit and integration tests
docs/                       Architecture, demo, and release documentation
```

## Documentation

- [`docs/demo.md`](docs/demo.md) — reproducible local demo
- [`docs/production-readiness.md`](docs/production-readiness.md) — release checklist and verified state
- [`docs/advanced-rag-agent.md`](docs/advanced-rag-agent.md) — advanced RAG/agent design
- [`docs/architecture.svg`](docs/architecture.svg) — architecture diagram

## Design goal

> **Gemini decides which capability is needed, LangGraph controls the execution loop, MCP provides the capability boundary, deterministic weather logic handles safety-sensitive scoring, and modular hybrid RAG provides grounded domain knowledge.**
