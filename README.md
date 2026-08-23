# MCP-Enabled Weather Intelligence System

An **MCP-first, local AI weather intelligence system** for Indian locations. The project demonstrates how a local Ollama model can discover and call weather capabilities through a real **Model Context Protocol (MCP) client/server boundary**, rather than directly importing application functions.

The goal is not to replace a normal weather app. The system demonstrates **LLM tool orchestration, real-time API integration, hybrid RAG retrieval, and decision-oriented weather analysis** while remaining free to run locally.

## Architecture

```text
User
  |
  v
Ollama local LLM
  |
  v
MCP Client
  |
  | MCP / stdio
  v
MCP Server
  |-------------------------------|
  v               v               v
Weather tools   Risk tool      Hybrid RAG
  |               |               |
Open-Meteo   Deterministic    Dense + BM25 + RRF
  |           risk scoring          |
  |---------------|---------------|
                  |
                  v
           Ollama final answer
```

## MCP Tools

- `get_weather` — current weather + 7-day forecast for an Indian location
- `assess_weather_risk` — deterministic activity-risk assessment using live forecast signals
- `search_weather` — hybrid vector/BM25 weather knowledge retrieval
- `sync_weather` — refresh weather data for configured locations
- `database_health` — check the configured database backend
- `ask_weather` — existing grounded RAG answer path, kept outside the agent to avoid nested LLM calls

The agent exposes `get_weather`, `assess_weather_risk`, and `search_weather` for normal tool calling. The risk tool performs deterministic scoring without another LLM request, keeping the local agent fast.

## Key Technologies

- **MCP v2** — tool discovery and protocol-based orchestration
- **Ollama** — local LLM inference; no paid LLM API required
- **Open-Meteo** — free weather data provider
- **OpenStreetMap Nominatim** — Indian location/geocoding resolution
- **Sentence Transformers** — dense retrieval/embeddings
- **BM25** — lexical retrieval
- **Reciprocal Rank Fusion (RRF)** — hybrid retrieval ranking
- **PostgreSQL/pgvector** — persistence and vector search
- **Python** — application and MCP implementation

## Local Setup

### 1. Clone and enter the repository

```bash
git clone https://github.com/Atul1127/MCP-Enabled-Weather-Intelligence-System.git
cd MCP-Enabled-Weather-Intelligence-System
```

### 2. Create the virtual environment

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Make sure Ollama is running

The default local model is:

```text
llama3.2:3b
```

You can change the agent model without editing code:

```bash
# Git Bash
export WEATHER_AGENT_MODEL=qwen3:4b
```

The RAG answer path can be configured independently with `WEATHER_LLM_MODEL`.

### 5. Run the MCP smoke test

```bash
python mcp_client.py
```

This starts the MCP server through stdio, initializes a real MCP client session, discovers the tools, and performs a sample Kolkata weather call.

### 6. Run the local agent

```bash
python weather_agent.py "What is the current weather in Kolkata?"
```

Risk-oriented example:

```bash
python weather_agent.py "Is Kolkata suitable for outdoor activities tomorrow? Explain the main risks."
```

Comparison example:

```bash
python weather_agent.py "Compare the current weather in Kolkata and Delhi."
```

### 7. Run tests

```bash
python -m pytest -q
```

The MCP integration test can be run independently with:

```bash
python -m pytest tests/test_mcp_client.py -q
```

## Docker

Docker Compose provides the existing Flask/RAG API and PostgreSQL + pgvector database. Ollama remains on the host so the local model can be reused without putting model weights into the image.

```bash
docker compose up --build
```

The API is available on port `8000` and uses `llama3.2:3b` by default. The MCP agent is normally run locally with `python weather_agent.py` so it can launch the MCP server over stdio.

## Performance Design

The local agent is intentionally bounded for practical laptop inference:

- small local model by default
- maximum two agent rounds
- MCP tool schemas are discovered once per session
- independent MCP tool calls are executed concurrently
- model is kept alive for repeated requests
- deterministic risk scoring avoids an extra LLM call
- RAG is lazily imported by the MCP server, so simple weather/risk requests do not initialize the embedding model
- `ask_weather` is not exposed to the agent because it would create a nested LLM call

For a simple weather question, the intended flow is:

```text
Ollama tool selection
        -> MCP get_weather
        -> weather API
        -> Ollama final synthesis
```

For a decision question:

```text
Ollama tool selection
        -> MCP assess_weather_risk
        -> live forecast + deterministic scoring
        -> Ollama final synthesis
```

## RAG Pipeline

The existing weather knowledge pipeline combines:

```text
Query
  |
  +--> Dense retrieval
  |
  +--> BM25 retrieval
  |
  +--> Reciprocal Rank Fusion
  |
  v
Relevant weather evidence
  |
  v
Local Ollama synthesis
```

RAG is a supporting knowledge layer; **MCP is the primary integration/orchestration concept**.

## Cost

The development stack is designed for **₹0 API cost**:

- local Ollama inference
- MCP Python SDK
- Open-Meteo free weather API
- local retrieval components
- local application runtime

No OpenAI/Gemini API key is required.

## Project Structure

```text
mcp_server.py                 MCP server and weather/RAG tools
mcp_client.py                 MCP stdio client and tool discovery
weather_agent.py              Ollama -> MCP agent loop
weather_client.py             Indian location + weather data client
rag_service.py                Hybrid weather retrieval/RAG
notebooks/                    Weather embedding/ingestion work
scripts/                      Data ingestion utilities
resources/                    Ingestion configuration
sql/                          Weather/database schema
templates/                    Flask UI templates
tests/                        API, retrieval, RAG and MCP tests
Dockerfile                    Flask/RAG container image
docker-compose.yml            PostgreSQL + Flask/RAG local stack
```

## Project Focus

The project is intentionally focused on one core engineering problem:

> **How can a local LLM safely discover and orchestrate external weather capabilities through MCP to produce useful, evidence-grounded answers?**

Weather is the demonstration domain; the MCP client/server architecture is the primary reusable engineering concept.
