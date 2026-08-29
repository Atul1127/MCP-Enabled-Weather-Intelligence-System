# Portfolio Summary

## One-line summary

An MCP-first weather intelligence platform that combines Gemini agentic tool calling, LangGraph orchestration, deterministic weather-risk logic, and grounded hybrid RAG for Indian locations.

## Resume bullets

- Built an MCP-enabled weather intelligence agent using Gemini tool calling and LangGraph to route bounded multi-step workflows across live weather, forecasts, alerts, risk assessment, and grounded knowledge retrieval.
- Implemented hybrid RAG with dense retrieval, BM25, confidence-aware reciprocal-rank fusion, reranking, and context compression while keeping retrieved evidence distinct from live weather data.
- Added production hardening including deterministic input/MCP validation, bounded tool outputs, prompt-injection defense-in-depth, non-root Docker execution, dropped capabilities, read-only filesystem, and readiness checks.
- Centralized Gemini text, structured-output, and tool-calling through a shared provider with retry/fallback handling and explicit model configuration.
- Built evaluation suites covering retrieval, RAG, agent routing, tool arguments, evidence sufficiency, latency, and end-to-end behavior; the latest recorded 16-case agent evaluation achieved 100% task success with zero infrastructure failures.

## Tech stack

Python 3.13 · Gemini API · LangGraph · MCP · Flask · Open-Meteo · BM25 · Sentence Transformers · Docker · PostgreSQL/Lakebase support · pytest · GitHub Actions

## Engineering highlights

The project deliberately separates deterministic safety-sensitive weather intelligence from probabilistic LLM generation. MCP provides the capability boundary, LangGraph controls execution, the evidence layer preserves provenance, and RAG supplies domain knowledge without becoming the source of live-weather claims.
