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


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _project_python() -> str:
    """Use the project virtualenv even when Chainlit itself is global."""
    configured = os.environ.get("WEATHER_PYTHON")
    if configured and os.path.isfile(configured):
        return configured

    current = os.path.abspath(sys.executable)
    if os.path.normcase(os.path.dirname(current)).endswith(
        os.path.normcase(os.path.join(".venv", "Scripts"))
    ):
        return current

    venv_python = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    if os.path.isfile(venv_python):
        return venv_python

    return current


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
    """Run the proven CLI agent in an isolated project-venv process."""
    agent_path = os.path.join(PROJECT_ROOT, "agent.py")
    python = _project_python()
    env = os.environ.copy()
    env["WEATHER_PYTHON"] = python

    completed = subprocess.run(
        [python, agent_path, query],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=int(os.environ.get("CHAINLIT_AGENT_TIMEOUT", "120")),
        env=env,
    )

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()

    if completed.returncode != 0:
        detail = stderr or stdout or f"agent exited with code {completed.returncode}"
        raise RuntimeError(detail[-4000:])

    trace_match = re.search(r"trace_id=([A-Za-z0-9_-]+)", stdout)
    trace_id = trace_match.group(1) if trace_match else None
    answer = stdout[: trace_match.start()].rstrip() if trace_match else stdout

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
