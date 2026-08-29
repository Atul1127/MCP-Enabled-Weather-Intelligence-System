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

Expected healthy responses have `status: "ok"` and `status: "ready"` respectively.

## 2. Run the agent from the CLI

```bash
python agent.py "What weather conditions are typically associated with heavy rainfall?"
```

For a live-weather example:

```bash
python agent.py "What is the current weather in Mumbai and is it risky for outdoor activity?"
```

## 3. Exercise the HTTP API

Current weather:

```bash
curl -X POST http://localhost:8000/weather/current \
  -H "Content-Type: application/json" \
  -d '{"city":"Mumbai"}'
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

## 4. Run evaluation

Unit/integration tests:

```bash
python -m pytest -q
```

Agent E2E evaluation:

```bash
python -m evaluation.agent_e2e_eval
```

Agent benchmark:

```bash
python -m evaluation.agent_benchmark
```

## 5. Observability

Trace output is written to the configured `WEATHER_TRACE_PATH`. After an agent run, inspect a trace with:

```bash
python evaluation/trace_report.py <trace_id>
```

## Important security note

Never paste `GEMINI_API_KEY` into source files, screenshots, chat messages, Git history, or committed configuration. If a key has been exposed, revoke/rotate it and replace the local environment value before rerunning the demo.
