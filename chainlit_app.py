"""Chainlit UI for the Indian Weather Intelligence Agent."""

from __future__ import annotations

import asyncio

import chainlit as cl

from agent import MODEL, run_agent
from llm_provider import provider_name


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(
        content=(
            "## Indian Weather Intelligence\n\n"
            "Ask about current weather, forecasts, weather risks, alerts, "
            "or weather knowledge.\n\n"
            f"LLM: `{provider_name()} / {MODEL}`"
        )
    ).send()


def _run_agent_isolated(query: str) -> dict:
    """Run the MCP stdio agent on its own asyncio event loop.

    Chainlit runs on an AnyIO task group. The MCP stdio transport also owns
    an AnyIO task group, and sharing the Chainlit event-loop lifecycle can
    surface the generic `TaskGroup` exception in the UI even though the same
    agent works correctly from the CLI. Running the complete agent lifecycle
    in a worker thread gives the MCP subprocess its own asyncio/AnyIO scope.
    """
    return asyncio.run(run_agent(query))


@cl.on_message
async def on_message(message: cl.Message) -> None:
    query = (message.content or "").strip()
    if not query:
        await cl.Message(content="Please enter a weather question.").send()
        return

    try:
        result = await asyncio.to_thread(_run_agent_isolated, query)
        answer = str(result.get("answer") or "No answer was returned.")
        trace_id = result.get("trace_id")
        if trace_id:
            answer = f"{answer}\n\n`trace_id={trace_id}`"
        await cl.Message(content=answer).send()
    except Exception as exc:
        await cl.Message(content=f"I couldn't process that request: {exc}").send()
