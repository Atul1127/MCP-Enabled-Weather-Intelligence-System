"""Chainlit UI for the Indian Weather Intelligence Agent."""

from __future__ import annotations

# Chainlit must always use Gemini for the UI. The CLI/API can still select
# another provider explicitly via WEATHER_LLM_PROVIDER, but the chat demo is
# intentionally pinned to the production cloud LLM path.
import os
os.environ["WEATHER_LLM_PROVIDER"] = "gemini"

import chainlit as cl

from agent import run_agent
from llm_provider import model_name, provider_name


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(
        content=(
            "## Indian Weather Intelligence\n\n"
            "Ask about current weather, forecasts, weather risks, alerts, "
            "or weather knowledge.\n\n"
            f"LLM: `{provider_name()} / {model_name()}`"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    query = (message.content or "").strip()
    if not query:
        await cl.Message(content="Please enter a weather question.").send()
        return

    try:
        result = await run_agent(query)
        answer = str(result.get("answer") or "No answer was returned.")
        trace_id = result.get("trace_id")
        if trace_id:
            answer = f"{answer}\n\n`trace_id={trace_id}`"
        await cl.Message(content=answer).send()
    except Exception as exc:
        await cl.Message(
            content=f"I couldn't process that request: {exc}"
        ).send()
