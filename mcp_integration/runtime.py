"""One long-lived MCP server session per worker process.

Starting the stdio MCP server costs a full Python start-up plus heavy imports.
Doing that on every HTTP request starves small instances (Render's free plan
has 0.1 CPU), so health checks time out and the platform restarts the
instance mid-request. This module starts the server once, keeps its session
open on a dedicated event-loop thread, and lets request threads run coroutines
against it.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Awaitable, Callable

from .mcp_client import connect

logger = logging.getLogger("weather-rag.mcp-runtime")

_START_TIMEOUT = float(os.environ.get("WEATHER_MCP_START_TIMEOUT", "90"))


class MCPRuntime:
    """Owns an event loop thread and a reusable MCP ClientSession."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: Any = None
        self._holder: asyncio.Future[Any] | None = None
        self._stop: asyncio.Event | None = None
        self._owner_pid: int | None = None
        self._start_lock: asyncio.Lock | None = None

    # -- loop -------------------------------------------------------------
    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        # Threads do not survive fork(); a forked worker must build its own loop.
        if self._loop is None or self._owner_pid != os.getpid() or not (self._thread and self._thread.is_alive()):
            loop = asyncio.new_event_loop()
            thread = threading.Thread(target=loop.run_forever, name="mcp-runtime", daemon=True)
            thread.start()
            self._loop, self._thread, self._owner_pid = loop, thread, os.getpid()
            self._session = self._holder = self._stop = self._start_lock = None
        return self._loop

    # -- session ----------------------------------------------------------
    async def _hold_session(self, ready: asyncio.Future[Any]) -> None:
        """Open the session and keep it open until stopped or the server dies."""
        self._stop = asyncio.Event()
        try:
            async with connect() as session:
                if not ready.done():
                    ready.set_result(session)
                await self._stop.wait()
        except BaseException as exc:  # surfaced to the waiting starter, then logged
            if not ready.done():
                ready.set_exception(exc)
            elif not isinstance(exc, asyncio.CancelledError):
                logger.warning("MCP session closed unexpectedly: %s", exc)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise

    async def _get_session(self) -> Any:
        if self._start_lock is None:
            self._start_lock = asyncio.Lock()
        async with self._start_lock:  # concurrent first requests share one server
            return await self._start_session_locked()

    async def _start_session_locked(self) -> Any:
        if self._session is not None and self._holder is not None and not self._holder.done():
            return self._session
        ready: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._holder = asyncio.ensure_future(self._hold_session(ready))
        self._session = await asyncio.wait_for(ready, timeout=_START_TIMEOUT)
        logger.info("MCP server session started (pid %s)", os.getpid())
        return self._session

    def _reset(self) -> None:
        if self._stop is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        self._session = None

    # -- public API -------------------------------------------------------
    def warm_up(self) -> None:
        """Start the MCP server in the background (e.g. right after a worker forks)."""
        with self._lock:
            loop = self._ensure_loop()
        asyncio.run_coroutine_threadsafe(self._get_session(), loop)

    def run(self, work: Callable[[Any], Awaitable[Any]], *, timeout: float) -> Any:
        """Run ``work(session)`` on the runtime loop and wait for its result.

        If the session is broken (server crashed), it is restarted once.
        """
        with self._lock:
            loop = self._ensure_loop()

        async def call() -> Any:
            session = await self._get_session()
            try:
                return await work(session)
            except Exception as exc:
                if _looks_like_dead_session(exc):
                    self._reset()
                    session = await self._get_session()
                    return await work(session)
                raise

        future = asyncio.run_coroutine_threadsafe(call(), loop)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            future.cancel()
            raise TimeoutError(f"Agent request exceeded {timeout:g}s") from None

    def close(self) -> None:
        with self._lock:
            self._reset()
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._loop.stop)


def _looks_like_dead_session(exc: BaseException) -> bool:
    names = {type(e).__name__ for e in _exception_chain(exc)}
    return bool(names & {"ClosedResourceError", "BrokenResourceError", "EndOfStream", "BrokenPipeError", "ConnectionResetError"})


def _exception_chain(exc: BaseException):
    seen: set[int] = set()
    stack = [exc]
    while stack:
        item = stack.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        yield item
        stack.extend(getattr(item, "exceptions", ()) or ())
        for linked in (item.__cause__, item.__context__):
            if linked is not None:
                stack.append(linked)


runtime = MCPRuntime()
