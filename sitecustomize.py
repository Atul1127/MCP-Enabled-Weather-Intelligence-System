"""Runtime compatibility shim for google-genai SDK versions."""

from __future__ import annotations

import inspect

try:
    from google.genai import types

    descriptor = types.Part.__dict__.get("from_function_response")
    factory = types.Part.from_function_response

    if "id" not in inspect.signature(factory).parameters:
        if isinstance(descriptor, classmethod):
            original = descriptor.__func__

            def _compatible_from_function_response(cls, **kwargs):
                # Some installed google-genai versions expose FunctionResponse.id
                # but do not expose id on Part.from_function_response(). Preserve
                # every supported argument and drop only the unsupported keyword.
                kwargs.pop("id", None)
                return original(cls, **kwargs)

            types.Part.from_function_response = classmethod(_compatible_from_function_response)
        else:
            original = factory

            def _compatible_from_function_response(**kwargs):
                kwargs.pop("id", None)
                return original(**kwargs)

            types.Part.from_function_response = staticmethod(_compatible_from_function_response)
except Exception:
    # Never prevent application startup because of an optional SDK shim.
    pass
