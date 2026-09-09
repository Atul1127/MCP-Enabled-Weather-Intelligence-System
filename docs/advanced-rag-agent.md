# Gemini Agent + LangGraph + Modular RAG

The main branch is a Gemini-only, MCP-first weather intelligence system with LangGraph orchestration and a modular RAG stack.

## Architecture

```text
User
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
  |              +--> live weather
  |              +--> forecast
  |              +--> hazard detection
  |              +--> activity risk
  |              +--> weather knowledge retrieval
  |                         |
  |                         v
  |                    RAGPipeline
  |                         |
  |                         +--> query analysis
  |                         +--> metadata filtering
  |                         +--> full-text retrieval
  |                         +--> optional pgvector retrieval
  |                         +--> simple top-k selection
  |                         +--> bounded context + citations
  |                         |
  |                         v
  +-------------------- Unified Evidence
                              |
                              v
                       Gemini Synthesizer
                              |
                     structured response
                              |
                     citation validation
                              |
                              v
                            Answer
```

The MCP server is the capability boundary. LangGraph owns reasoning and orchestration. RAG owns retrieval and evidence preparation. The final Gemini synthesizer combines live MCP evidence with RAG evidence and returns a validated structured contract.

## Security boundary

Tool arguments are validated before MCP execution, and untrusted MCP results are bounded by type, size, and nesting checks. The agent also applies deterministic prompt-injection signal checks at the user-input boundary. These checks are defense-in-depth; they are not a complete prompt-injection solution.

## RAG backend

The default deployment uses PostgreSQL full-text retrieval. Optional dense retrieval uses pgvector when explicitly enabled. Retrieved documents are filtered, ranked, bounded, and formatted with stable citations before reaching the synthesizer.

The default runtime does not require the optional transformer stack.

## Gemini

Gemini is the only LLM provider. Generation, structured output, and tool-calling requests pass through the shared LLM provider gateway.

## Evaluation

Run the repository tests first:

```bash
python -m pytest -q
```

Then run the retrieval and agent benchmarks as needed. Live evaluations consume Gemini quota and should be run deliberately.

## Observability

Agent, MCP, retrieval, context formatting, and synthesis stages emit trace events with a shared trace ID.
