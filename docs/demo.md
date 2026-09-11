# Demo Guide

This guide provides a short, reproducible walkthrough for the MCP-Enabled Weather Intelligence System.

## 1. Start the API

Create `.env` from `.env.example` and set a valid `GEMINI_API_KEY`, then run:

```bash
docker compose up --build -d
```

Verify:

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz
```

Open the dashboard at:

```text
http://localhost:8000/
```

## 2. Run the agent from the CLI

Knowledge/RAG example:

```bash
python agent.py "What weather conditions are typically associated with heavy rainfall?"
```

Live-weather example:

```bash
python agent.py "What is the current weather in Mumbai and is it risky for outdoor activity?"
```

## 3. Exercise the HTTP API

Current weather:

```bash
curl -X POST http://localhost:8000/weather/current \
  -H "Content-Type: application/json" \
  -d '{"location":"Mumbai"}'
```

Weather knowledge / RAG:

```bash
curl -X POST http://localhost:8000/weather/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What conditions are associated with heavy rainfall?"}'
```

Full agent workflow:

```bash
curl -X POST http://localhost:8000/weather/agent \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the current weather in Mumbai and what outdoor risks should I consider?"}'
```

## 4. Production deployment

The application is deployed on Railway:

https://mcp-enabled-weather-intelligence-system-production.up.railway.app/

The same probes are available remotely:

```bash
curl https://mcp-enabled-weather-intelligence-system-production.up.railway.app/healthz
curl https://mcp-enabled-weather-intelligence-system-production.up.railway.app/readyz
```

## 5. Run evaluation

Unit/integration tests:

```bash
python -m pytest -q
```

Balanced 100-case live agent evaluation:

```bash
WEATHER_STRESS_LIMIT=100 WEATHER_STRESS_REPORT=/tmp/stress_report_100.json python -m evaluation.stress_eval
```

The stress suite covers current-weather, forecast, alert, activity-risk, and weather-knowledge cases across multiple Indian cities. It reports task success, tool-selection accuracy, argument accuracy for evaluable tool cases, provider quota failures, and latency percentiles.

Agent benchmark:

```bash
python -m evaluation.agent_benchmark
```

## 6. Observability

Trace output is written to the configured `WEATHER_TRACE_PATH`. After an agent run, inspect a trace with:

```bash
python evaluation/trace_report.py <trace_id>
```

## Important security note

Never paste `GEMINI_API_KEY` into source files, screenshots, chat messages, Git history, or committed configuration. If a key has been exposed, revoke/rotate it and replace the local environment value before rerunning the demo.
