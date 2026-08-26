"""Chainlit UI for the Indian Weather Intelligence Agent."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys

import chainlit as cl

from agent import MODEL
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


def _run_agent_process(query: str) -> tuple[str, str | None]:
    """Run the proven CLI agent in a completely isolated process.

    Chainlit and the MCP stdio transport both use AnyIO task groups. Keeping
    the MCP lifecycle inside a child process avoids sharing Chainlit's async
    cancellation/task-group scope with the MCP subprocess transport.
    """
    agent_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent.py")
    completed = subprocess.run(
        [sys.executable, agent_path, query],
        cwd=os.path.dirname(agent_path),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=int(os.environ.get("CHAINLIT_AGENT_TIMEOUT", "120")),
        env=os.environ.copy(),
    )

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()

    if completed.returncode != 0:
        detail = stderr or stdout or f"agent exited with code {completed.returncode}"
        raise RuntimeError(detail[-4000:])

    trace_match = re.search(r"trace_id=([A-Za-z0-9_-]+)", stdout)
    trace_id = trace_match.group(1) if trace_match else None

    if trace_match:
        answer = stdout[: trace_match.start()].rstrip()
    else:
        answer = stdout

    if not answer:
        raise RuntimeError("Agent returned an empty answer")

    return answer, trace_id


@cl.on_message
async def on_message(message: cl.Message) -> None:
    query = (message.content or "").strip()
    if not query:
        await cl.Message(content="Please enter a weather question.").send()
        return

    try:
        answer, trace_id = await asyncio.to_thread(_run_agent_process, query)
        if trace_id:
            answer = f"{answer}\n\n`trace_id={trace_id}`"
        await cl.Message(content=answer).send()
    except subprocess.TimeoutExpired:
        await cl.Message(
            content="The weather agent timed out. Please retry the request."
        ).send()
    except Exception as exc:
        await cl.Message(content=f"I couldn't process that request: {exc}").send()
