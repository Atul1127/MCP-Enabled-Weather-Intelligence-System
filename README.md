# MCP-Enabled Weather Intelligence System

An **MCP-first weather intelligence platform for Indian locations** combining live weather data, deterministic risk intelligence, grounded PostgreSQL RAG, Gemini tool calling, LangGraph orchestration, security boundaries, observability, and evaluation.

## 🚀 Live Demo

**Production dashboard:** https://mcp-enabled-weather-intelligence-system-production.up.railway.app/

**Health:** https://mcp-enabled-weather-intelligence-system-production.up.railway.app/healthz  
**Readiness:** https://mcp-enabled-weather-intelligence-system-production.up.railway.app/readyz

The deployed service exposes the same dashboard and API used by the local Docker setup.

## What it demonstrates

- **Agentic orchestration:** LangGraph controls a bounded workflow while Gemini handles tool selection and structured synthesis.
- **MCP capability boundary:** Weather and knowledge capabilities are exposed through protocol-based tools with an explicit model-facing allowlist.
- **Grounded RAG:** PostgreSQL full-text retrieval is the default; pgvector dense retrieval is optional.
- **Deterministic intelligence:** Hazard detection and activity-risk scoring are computed separately from LLM generation.
- **Evidence-first answers:** Live weather, forecasts, alerts, risk results, and retrieved knowledge remain typed and auditable before synthesis.
- **Production hardening:** Input validation, prompt-injection checks, MCP allowlists, bounded observations, retries, timeouts, non-root containers, dropped capabilities, read-only filesystems, and readiness checks.
- **Evaluation:** Unit/integration tests plus retrieval, RAG, agent, answer-quality, and balanced live stress evaluation suites.
- **Interactive dashboard:** Current conditions, hourly forecast, 7-day forecast, application-level forecast risk, alerts, and an AI Weather Agent.

## Architecture

```text
User / Dashboard / HTTP API
            |
            v
      LangGraph WeatherAgent
            |
      Router -> Planner -> Decomposer
            |
            v
   Reasoner <-> Verifier
            |
            v
      MCP Executor
       /    |     \
      /     |      \
 live   risk/alerts   weather knowledge
 tools     tools          |
                          v
                  PostgreSQL RAG
                  +-- FTS (default)
                  +-- optional pgvector
                  +-- simple top-k
                  +-- bounded context + citations
            |
            v
      Unified Evidence
            |
            v
      Gemini Synthesizer
            |
      citation validation
            |
            v
          Answer
```

### RAG flow

```text
Query
  -> query analysis
  -> PostgreSQL full-text retrieval
  -> optional pgvector retrieval
  -> simple hybrid fusion when dense retrieval is enabled
  -> top-k selection
  -> bounded citation-ready context
  -> grounded synthesis
```

The production default keeps dense retrieval disabled, so the normal API image does not need PyTorch or Sentence Transformers.

## Dashboard

The web interface is organized around the information most useful to a weather decision:

1. Location search
2. Current weather
3. Next 12 hours
4. Forecast risk
5. 7-day forecast
6. Alerts & advisories
7. AI Weather Agent

The dashboard is responsive and includes explicit loading, error, and empty states.

## Gemini configuration

Gemini is the only LLM provider. Generation, structured output, and tool-calling requests pass through the shared `llm_provider.py` gateway.

```bash
GEMINI_API_KEY=your-key
GEMINI_MODEL=gemini-3.6-flash
GEMINI_FALLBACK_MODELS=gemini-3.5-flash-lite
GEMINI_THINKING_LEVEL=low
GEMINI_MAX_OUTPUT_TOKENS=700
```

The provider distinguishes quota exhaustion from transient failures: exhausted model quotas are not repeatedly retried, while transient failures can be retried before moving to a fallback model.

Never commit a real API key. Rotate any credential that has been exposed.

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

Open `http://localhost:8000/` for the dashboard.

### Docker Compose

```bash
docker compose up --build -d
```

Verify:

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz
```

The default Docker stack uses **PostgreSQL + pgvector** for persistence and RAG. Sparse/full-text retrieval is the normal path; dense retrieval is opt-in with `WEATHER_RAG_DENSE=1` and the optional ML dependencies.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Dashboard |
| `/healthz` | GET | Liveness |
| `/readyz` | GET | Dependency readiness |
| `/weather/current` | POST | Current weather + forecast data |
| `/weather/alerts` | POST | Deterministic forecast hazard detection |
| `/weather/ask` | POST | Grounded weather knowledge / RAG |
| `/weather/agent` | POST | Full LangGraph + MCP agent |
| `/weather/sync` | POST | Protected persistence synchronization |

Example:

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
| `search_weather` | Weather knowledge retrieval |
| `ask_weather` | Weather knowledge retrieval alias |
| `sync_weather` | Persistence synchronization; excluded from the model-facing allowlist |
| `database_health` | Persistence health check |

## Evaluation

Run the full automated test suite:

```bash
python -m pytest -q
```

The current release has **122 tests passing** locally.

Run the **balanced 100-case live stress evaluation** deliberately because it consumes Gemini quota:

```bash
WEATHER_STRESS_LIMIT=100 WEATHER_STRESS_REPORT=/tmp/stress_report_100.json python -m evaluation.stress_eval
```

The stress suite covers current-weather, forecast, alert, activity-risk, and weather-knowledge cases across multiple Indian cities. It reports task success, tool-selection accuracy, argument accuracy for evaluable tool cases, provider quota failures, and latency percentiles.

## Observability

Agent, MCP, retrieval, context formatting, and synthesis stages emit JSONL trace events with a shared trace ID. No prompts or secrets are intentionally written to traces.

```bash
python evaluation/trace_report.py <trace_id>
```

Set `WEATHER_TRACE_PATH` to change the trace destination.

## Security

The system applies defense-in-depth controls at the user, planner, MCP, and synthesis boundaries:

- Unicode normalization and invisible/control-character checks
- Prompt-injection signal detection
- Location, query, top-k, tool-name, and tool-argument validation
- Explicit MCP tool allowlist
- Bounded observation size/depth
- Tool timeouts, retry policy, and duplicate-call coalescing
- Untrusted-data handling in synthesis prompts
- Protected synchronization endpoint
- Non-root Docker user
- Dropped Linux capabilities
- `no-new-privileges`
- Read-only container root filesystem

These controls reduce attack surface but are not a complete prompt-injection guarantee.

## Runtime dependency boundary

The normal API runtime keeps optional dense-RAG ML dependencies (`torch` and `sentence-transformers`) out of the default installation. They live in `requirements-rag-ml.txt` and are required only when dense retrieval is explicitly enabled.

## Project structure

```text
weather_agent_core/   LangGraph agent, planning, verification, MCP execution, security
rag/                  PostgreSQL retrieval, optional dense retrieval, citations, context
weather_client.py     Open-Meteo/Nominatim weather integration
mcp_server.py         MCP weather capability server
app.py                Flask API + dashboard
lakebase.py           PostgreSQL/Lakebase persistence abstraction
evaluation/           Retrieval, RAG, agent, answer, stress, and trace evaluation
tests/                Automated regression and architecture tests
docs/                 Architecture, deployment, demo, and production documentation
db/                   PostgreSQL + pgvector initialization
Dockerfile            Production container image
docker-compose.yml    Local production-style stack
```

## Documentation

- [`docs/demo.md`](docs/demo.md) — reproducible local/API demo
- [`docs/deployment.md`](docs/deployment.md) — Docker and hosted deployment guidance
- [`docs/advanced-rag-agent.md`](docs/advanced-rag-agent.md) — agent and RAG architecture
- [`docs/production-readiness.md`](docs/production-readiness.md) — release verification checklist
- [`docs/portfolio-summary.md`](docs/portfolio-summary.md) — portfolio/resume summary

## License

See the repository license file if present.
