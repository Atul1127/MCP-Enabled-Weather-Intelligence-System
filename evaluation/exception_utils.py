"""Exception classification for end-to-end evaluation."""
from __future__ import annotations


def root_exception(exc: BaseException) -> BaseException:
    """Return the deepest useful exception, including nested ExceptionGroups."""
    if isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        return root_exception(exc.exceptions[0])
    return exc


def classify_exception(exc: BaseException) -> str:
    """Classify infrastructure failures separately from agent failures."""
    root = root_exception(exc)
    message = str(root).lower()

    if (
        "resource_exhausted" in message
        or "quota" in message
        or "429" in message
    ):
        return "quota_exhausted"

    if (
        "unauthenticated" in message
        or "api key" in message
        or "authentication" in message
        or "permission_denied" in message
    ):
        return "authentication_failure"

    if (
        "connection" in message
        or "timeout" in message
        or "network" in message
    ):
        return "network_failure"

    return type(root).__name__
