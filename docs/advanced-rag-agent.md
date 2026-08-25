# Advanced local agent + RAG

The upgrade branch turns the weather project into a local-first agentic RAG system.

## Architecture

```text
User
  |
  v
ReAct Agent (Ollama)
  |
  +--> MCP live weather tools
  |      +--> current forecast
  |      +--> alerts
  |      +--> activity risk
  |
  +--> MCP retrieval tools
         |
         +--> query expansion
         +--> metadata filtering
         +--> dense retrieval
         +--> BM25
         +--> Reciprocal Rank Fusion
         +--> cross-encoder reranking
         +--> grounded generation

All stages emit trace events to observability/traces.jsonl.
```

## Local run

Start Ollama and make sure the configured model is available (default: `llama3.2:3b`).
Then run:

```bash
python agent.py "Compare Kolkata and Mumbai for outdoor activities tomorrow"
```

The agent can make multiple MCP tool calls, observe their results, and continue
reasoning until it has enough evidence or reaches `WEATHER_AGENT_MAX_ROUNDS`.

## Advanced RAG

`advanced_rag.py` adds multi-query expansion, optional location/state filters,
dense retrieval, BM25, RRF fusion, cross-encoder reranking, bounded context
construction, source citations, and structured trace IDs.

## Evaluation

Run the local answer-quality benchmark with:

```bash
python evaluation/evaluate_rag_local.py
```

The evaluator exercises the real RAG path and uses the local Ollama model as a
strict judge for faithfulness, relevance, and citation quality.

The existing MCP tool-selection evaluator remains available in
`evaluation/evaluate_agent.py`.

## Observability

Set `WEATHER_TRACE_PATH` to change the JSONL destination. Each agent/RAG run
gets a trace ID and emits spans for reasoning, tool execution, retrieval,
query expansion, and generation. No external telemetry service is required.
