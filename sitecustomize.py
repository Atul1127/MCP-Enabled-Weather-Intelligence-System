"""Runtime compatibility shim for google-genai SDK versions."""

from __future__ import annotations

import inspect

try:
    from google.genai import types

    _factory = types.Part.from_function_response
    _params = inspect.signature(_factory).parameters

    if "id" not in _params:
        def _compatible_from_function_response(*, name: str, response: dict, id: str | None = None, parts=None):
            # The installed SDK helper does not accept `id`. Use its supported
            # constructor shape rather than passing the unsupported keyword.
            return _factory(name=name, response=response, parts=parts)

        # Keep the descriptor shape expected by Part's classmethod API.
        types.Part.from_function_response = classmethod(_compatible_from_function_response)
except Exception:
    # Never prevent application startup because of an optional SDK shim.
    pass
