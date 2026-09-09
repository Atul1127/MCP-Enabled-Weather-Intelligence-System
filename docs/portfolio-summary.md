# Portfolio Summary

## Live deployment

https://mcp-enabled-weather-intelligence-system-production.up.railway.app/

## One-line summary

An MCP-first weather intelligence platform combining Gemini tool calling, LangGraph orchestration, deterministic weather-risk logic, and grounded PostgreSQL RAG for Indian locations.

## Resume bullets

- Built an MCP-enabled weather intelligence agent using Gemini tool calling and LangGraph to orchestrate bounded workflows across live weather, forecasts, alerts, risk assessment, and knowledge retrieval.
- Implemented grounded PostgreSQL RAG with full-text retrieval, optional pgvector dense retrieval, simple hybrid fusion, bounded context, and stable citations.
- Added production hardening with deterministic input/MCP validation, prompt-injection defense-in-depth, bounded tool outputs, retries/timeouts, non-root Docker execution, dropped capabilities, and read-only filesystems.
- Centralized Gemini text, structured-output, and tool-calling through a shared provider with quota-aware retry/fallback handling and explicit model configuration.
- Built evaluation suites covering retrieval, RAG, routing, tool arguments, evidence sufficiency, latency, and end-to-end behavior; the verified live agent evaluation achieved 16/16 task success with 100% tool-selection and argument accuracy.

## Tech stack

Python 3.13 · Gemini API · LangGraph · MCP · Flask · Open-Meteo · PostgreSQL · pgvector · Docker · pytest · GitHub Actions

## Engineering highlights

The project deliberately separates deterministic, safety-sensitive weather intelligence from probabilistic LLM generation. MCP provides the capability boundary, LangGraph controls execution, the evidence layer preserves provenance, and RAG supplies domain knowledge without becoming the source of live-weather claims.

The default runtime keeps the optional transformer stack out of the API image, while PostgreSQL remains the production persistence/RAG boundary.
