"""Chainlit UI for the Indian Weather Intelligence Agent."""

from __future__ import annotations

import chainlit as cl

from agent import run_agent


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(
        content=(
            "## Indian Weather Intelligence\n\n"
            "Ask about current weather, forecasts, weather risks, alerts, "
            "or weather knowledge."
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
