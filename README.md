# MCP-Enabled Weather Intelligence System

An **MCP-first weather intelligence platform for Indian locations** that combines live weather data, deterministic risk intelligence, hybrid RAG, Gemini tool calling, LangGraph orchestration, security boundaries, observability, and evaluation.

## Live Demo

🚀 **Live Application:** https://mcp-enabled-weather-intelligence-system-production.up.railway.app

## What this project demonstrates

- **Agentic orchestration:** Gemini selects capabilities while LangGraph controls a bounded execution loop.
- **MCP architecture:** Weather and retrieval capabilities are exposed through protocol-based tools with an explicit model-facing allowlist.
- **Grounded RAG:** PostgreSQL full-text retrieval is the default; optional dense retrieval uses pgvector.
- **Safety-oriented intelligence:** Deterministic hazard detection and activity-risk scoring are separated from LLM generation.
- **Evidence-first answers:** Live weather, forecasts, risks, alerts, and retrieved knowledge remain typed and auditable before synthesis.
- **Production controls:** Input checks, MCP argument validation, bounded tool results, read-only containers, dropped capabilities, and no-new-privileges.
- **Evaluation:** Retrieval, RAG, agent, answer-quality, and end-to-end benchmark suites.
- **Interactive dashboard:** Responsive weather dashboard with current conditions, hourly forecast, 7-day forecast, forecast risk, filtered alerts, and an AI Weather Agent.

## Dashboard

The web interface is designed around the most useful weather information first:

```text
Location Search
      ↓
Current Weather
      ↓
Hourly Forecast (next 12 hours)
      ↓
Forecast Risk
      ↓
7-Day Forecast
      ↓
Alerts & Advisories
      ↓
AI Weather Agent
```

### Dashboard capabilities

- **Current weather:** Temperature, feels-like temperature, humidity, wind, and resolved location.
- **Hourly forecast:** Next 12 hours with time, weather condition, temperature, precipitation probability, rainfall, and wind.
- **Forecast risk:** Application-level hazard severity with a compact risk indicator.
- **7-day forecast:** Daily weather icons and high/low temperatures.
- **Alerts & Advisories:** Forecast hazards grouped by date with severity filtering and progressive disclosure.
- **AI Weather Agent:** Natural-language weather questions grounded in live tools and retrieved weather knowledge.
- **Responsive UI:** Desktop and mobile layouts with accessible loading, error, and empty states.

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
       |                    PostgreSQL RAG
       |                         |
       |                         +--> full-text retrieval (default)
       |                         +--> optional pgvector dense retrieval
       |                         +--> simple top-k selection
       |                         +--> bounded context + citations
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

The provider distinguishes quota exhaustion from transient failures: exhausted model quotas are not retried against the same model, while transient provider failures may be retried before moving to a fallback model.

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

The default Docker deployment uses **PostgreSQL + pgvector** for the RAG store. Dense vector retrieval is disabled by default (`WEATHER_RAG_DENSE=0`) so the normal API image does not require the optional transformer stack. Enable it explicitly when the ML RAG dependencies are installed.

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
| `get_weather` | Current weather and 7-day forecast data |
| `get_forecast` | Specific future-day forecast |
| `get_weather_alerts` | Deterministic forecast-based hazard detection |
| `assess_weather_risk` | Deterministic activity-risk assessment |
| `search_weather` | Weather knowledge retrieval |
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

Run the balanced 100-case stress evaluation:

```bash
WEATHER_STRESS_LIMIT=100 WEATHER_STRESS_REPORT=/tmp/stress_report_100.json python -m evaluation.stress_eval
```

The stress suite records task success, tool-selection accuracy, argument accuracy for evaluable tool cases, and latency percentiles. Live evaluations consume Gemini quota and should be run deliberately.

## Runtime dependency boundary

The default API image keeps optional dense-RAG ML dependencies (`torch` and `sentence-transformers`) out of the runtime installation. They are listed in `requirements-rag-ml.txt` and loaded lazily only when dense retrieval is explicitly enabled.

This keeps the normal API image lightweight and avoids pulling the large ML dependency tree into the default runtime.

## Observability

Agent, MCP, retrieval, context formatting, and synthesis stages emit trace events with a shared trace ID. Inspect a trace with:

```bash
python evaluation/trace_report.py <trace_id>
```

Set `WEATHER_TRACE_PATH` to change the JSONL destination.

## Security
