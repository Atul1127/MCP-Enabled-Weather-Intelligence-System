"""Small runtime compatibility shim for the installed google-genai SDK.

Some google-genai releases expose FunctionResponse.id on the model but omit the
id keyword from Part.from_function_response(). The agent uses the documented
function-response id, so bridge that SDK-version mismatch without changing the
agent protocol or dropping Gemini 3 function-call IDs.
"""

from __future__ import annotations

import inspect

try:
    from google.genai import types

    _factory = types.Part.from_function_response
    if "id" not in inspect.signature(_factory).parameters:
        def _from_function_response(*, name: str, response: dict, id: str | None = None, parts=None):
            return types.Part(
                function_response=types.FunctionResponse(
                    id=id,
                    name=name,
                    response=response,
                    parts=parts,
                )
            )

        types.Part.from_function_response = classmethod(_from_function_response)
except Exception:
    # Never make Python startup fail because of an optional SDK compatibility hook.
    pass
