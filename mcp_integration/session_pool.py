"""Keep one MCP server process alive per worker and share it across requests.

Starting ``python -m mcp_integration.mcp_server`` for every agent request costs
a full Python start-up plus imports, and throws away everything the server
keeps in memory (RAG pipeline, retrieval cache, weather cache). This module
starts the server once, on a background event loop owned by a daemon thread,
and lets each request use that session from its own event loop.

Set ``WEATHER_MCP_PERSISTENT=0`` to go back to one server process per request
(useful when debugging, because the per-request mode also passes the trace ID
to the server process).
"""
from __future__ import annotations

import asyncio
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from .mcp_client import connect

STARTUP_TIMEOUT_SECONDS = float(os.environ.get("WEATHER_MCP_STARTUP_TIMEOUT", "60"))


def persistent_enabled() -> bool:
    return os.environ.get("WEATHER_MCP_PERSISTENT", "1").strip().lower() not in {"0", "false", "no", "off"}


class _SessionProxy:
    """Forward coroutine methods of a ClientSession to the loop that owns it."""

    def __init__(self, session: Any, loop: asyncio.AbstractEventLoop, pool: "_PersistentSession") -> None:
        self._session = session
        self._loop = loop
        self._pool = pool

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._session, name)
        if not asyncio.iscoroutinefunction(attr):
            return attr

        async def forward(*args: Any, **kwargs: Any) -> Any:
            future = asyncio.run_coroutine_threadsafe(attr(*args, **kwargs), self._loop)
            try:
                return await asyncio.wrap_future(future)
            except Exception:
                # A dead server process shows up as a transport error. Drop the
                # session so the next request starts a fresh server.
                if not self._pool.is_alive():
                    self._pool.reset()
                raise

        return forward


class _PersistentSession:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: Any = None
        self._stop: asyncio.Event | None = None
        self._task: asyncio.Task | None = None

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or not self._loop.is_running():
            loop = asyncio.new_event_loop()
            ready = threading.Event()

            def run() -> None:
                asyncio.set_event_loop(loop)
                loop.call_soon(ready.set)
                loop.run_forever()

            thread = threading.Thread(target=run, name="mcp-session-loop", daemon=True)
            thread.start()
            ready.wait()
            self._loop, self._thread = loop, thread
        return self._loop

    async def _open(self) -> tuple[Any, asyncio.Event, asyncio.Task]:
        """Runs on the background loop. The session lives inside one task so
        the stdio transport is entered and exited in the same task."""
        loop = asyncio.get_running_loop()
        ready: asyncio.Future = loop.create_future()
        stop = asyncio.Event()

        async def holder() -> None:
            try:
                async with connect() as session:
                    ready.set_result(session)
                    await stop.wait()
            except BaseException as exc:  # noqa: BLE001 - reported through `ready`
                if not ready.done():
                    ready.set_exception(exc)
                    return
                raise

        task = loop.create_task(holder())
        session = await ready
        return session, stop, task

    def is_alive(self) -> bool:
        return self._session is not None and self._task is not None and not self._task.done()

    def acquire(self) -> tuple[Any, asyncio.AbstractEventLoop]:
        with self._lock:
            loop = self._ensure_loop()
            if not self.is_alive():
                future = asyncio.run_coroutine_threadsafe(self._open(), loop)
                self._session, self._stop, self._task = future.result(timeout=STARTUP_TIMEOUT_SECONDS)
            return self._session, loop

    def reset(self) -> None:
        with self._lock:
            stop, loop = self._stop, self._loop
            self._session, self._stop, self._task = None, None, None
        if stop is not None and loop is not None and loop.is_running():
            loop.call_soon_threadsafe(stop.set)


_POOL = _PersistentSession()


@asynccontextmanager
async def agent_session(trace_id: str | None = None) -> AsyncIterator[Any]:
    """Yield an MCP session for one agent request.

    In persistent mode (the default) every request in this process shares one
    server. Otherwise a new server process is started for the request.
    """
    if not persistent_enabled():
        async with connect(trace_id=trace_id) as session:
            yield session
        return
    session, loop = await asyncio.to_thread(_POOL.acquire)
    yield _SessionProxy(session, loop, _POOL)


def shutdown() -> None:
    """Stop the shared server (mainly for tests)."""
    _POOL.reset()
