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
  |-----------------------------|
  v                             v
Weather tools              Hybrid RAG
  |                             |
Open-Meteo                Dense + BM25 + RRF
  |                             |
  |-----------------------------|
                 |
                 v
          Ollama final answer
```

## MCP Tools

- `get_weather` — current weather + 7-day forecast for an Indian location
- `search_weather` — hybrid vector/BM25 weather knowledge retrieval
- `sync_weather` — refresh weather data for configured locations
- `database_health` — check the configured database backend
- `ask_weather` — existing grounded RAG answer path, kept outside the agent to avoid nested LLM calls

The agent intentionally exposes only `get_weather` and `search_weather` for normal tool calling. This keeps local inference fast and prevents an unnecessary nested Ollama loop.

## Key Technologies

- **MCP** — tool discovery and protocol-based orchestration
- **Ollama** — local LLM inference; no paid LLM API required
- **Open-Meteo** — free weather data provider
- **OpenStreetMap Nominatim** — Indian location/geocoding resolution
- **Sentence Transformers** — dense retrieval/embeddings
- **BM25** — lexical retrieval
- **Reciprocal Rank Fusion (RRF)** — hybrid retrieval ranking
- **PostgreSQL/Lakebase** — existing persistence layer
- **Python** — application and MCP implementation

## Local Setup

### 1. Clone and enter the repository

```bash
git clone https://github.com/Atul1127/MCP-Enabled-Weather-Intelligence-System.git
cd MCP-Enabled-Weather-Intelligence-System
git checkout mcp-modernization
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

The default agent model is:

```text
llama3.2:3b
```

You can change it without editing code:

```bash
# Git Bash
export WEATHER_AGENT_MODEL=qwen3:4b
```

### 5. Run the MCP smoke test

```bash
python mcp_client.py
```

This starts the MCP server through stdio, initializes a real MCP client session, discovers the tools, and performs a sample Kolkata weather call.

### 6. Run the local agent

```bash
python weather_agent.py "What is the current weather in Kolkata?"
```

Example comparison:

```bash
python weather_agent.py "Compare the current weather in Kolkata and Delhi."
```

## Performance Design

The local agent is intentionally bounded for practical laptop inference:

- small local model by default
- maximum two agent rounds
- MCP tool schemas are discovered once per session
- independent MCP tool calls are executed concurrently
- model is kept alive for repeated requests
- `ask_weather` is not exposed to the agent because it would create a nested LLM call

For a simple weather question, the intended flow is:

```text
Ollama tool selection
        -> MCP get_weather
        -> weather API
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
prompts/                      RAG prompts
rag/                          Retrieval components
tests/                        API, retrieval, RAG and weather tests
```

## Project Focus

The project is intentionally focused on one core engineering problem:

> **How can a local LLM safely discover and orchestrate external weather capabilities through MCP to produce useful, evidence-grounded answers?**

Weather is the demonstration domain; the MCP client/server architecture is the primary reusable engineering concept.
