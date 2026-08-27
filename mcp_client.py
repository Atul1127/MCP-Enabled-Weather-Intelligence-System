"""Reusable MCP stdio client for the Indian Weather Intelligence server."""
from __future__ import annotations
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER_PATH = os.path.join(PROJECT_ROOT, "mcp_server.py")

def _python_executable() -> str:
    configured = os.environ.get("WEATHER_PYTHON")
    if configured and os.path.isfile(configured):
        return configured
    current = os.path.abspath(sys.executable)
    if os.path.normcase(os.path.dirname(current)).endswith(os.path.normcase(os.path.join(".venv", "Scripts"))):
        return current
    venv_python = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    return venv_python if os.path.isfile(venv_python) else current

@asynccontextmanager
async def connect(trace_id: str | None = None) -> AsyncIterator[ClientSession]:
    """Open an MCP stdio session using the project's dependency environment."""
    python = _python_executable()
    server_env = os.environ.copy()
    server_env.update({"WEATHER_PYTHON": python, "PYTHONUNBUFFERED": "1"})
    if trace_id:
        server_env["WEATHER_TRACE_ID"] = trace_id
    server_params = StdioServerParameters(command=python, args=[SERVER_PATH], env=server_env, cwd=PROJECT_ROOT)
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            try:
                await session.initialize()
            except Exception as exc:
                raise RuntimeError(f"MCP server failed to initialize using {python}: {exc}") from exc
            yield session

def _tool_schema(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "input_schema", None)
    if schema is None:
        schema = getattr(tool, "inputSchema", None)
    if hasattr(schema, "model_dump"):
        schema = schema.model_dump(by_alias=True, exclude_none=True)
    return {"type": "function", "function": {"name": tool.name, "description": tool.description or "", "parameters": schema or {"type": "object", "properties": {}}}}

async def discover_tools(session: ClientSession) -> list[dict[str, Any]]:
    response = await session.list_tools()
    tools = getattr(response, "tools", None)
    if tools is None:
        tools = []
        for item in response:
            if isinstance(item, tuple) and item and item[0] == "tools":
                tools.extend(item[1])
    return [_tool_schema(tool) for tool in tools]

async def discover_resources(session: ClientSession) -> list[dict[str, Any]]:
    """Discover concrete MCP resources and resource templates."""
    response = await session.list_resources()
    resources = getattr(response, "resources", None) or []
    templates_response = await session.list_resource_templates()
    templates = getattr(templates_response, "resource_templates", None) or []
    return [
        {
            "kind": "resource",
            "uri": str(getattr(item, "uri", "")),
            "name": str(getattr(item, "name", "")),
            "description": str(getattr(item, "description", "") or ""),
            "mime_type": getattr(item, "mime_type", None) or getattr(item, "mimeType", None),
        }
        for item in resources
    ] + [
        {
            "kind": "template",
            "uri_template": str(getattr(item, "uri_template", "") or getattr(item, "uriTemplate", "")),
            "name": str(getattr(item, "name", "")),
            "description": str(getattr(item, "description", "") or ""),
        }
        for item in templates
    ]

async def read_resource(session: ClientSession, uri: str) -> list[dict[str, Any]]:
    """Read an MCP resource and normalize its contents."""
    result = await session.read_resource(uri)
    contents = getattr(result, "contents", None) or []
    normalized = []
    for item in contents:
        normalized.append({
            "uri": str(getattr(item, "uri", uri)),
            "mime_type": getattr(item, "mime_type", None) or getattr(item, "mimeType", None),
            "text": getattr(item, "text", None),
            "blob": getattr(item, "blob", None),
        })
    return normalized

async def discover_prompts(session: ClientSession) -> list[dict[str, Any]]:
    """Discover reusable MCP prompt templates and their argument schemas."""
    response = await session.list_prompts()
    prompts = getattr(response, "prompts", None) or []
    return [
        {
            "name": str(getattr(item, "name", "")),
            "description": str(getattr(item, "description", "") or ""),
            "arguments": [
                {
                    "name": str(getattr(arg, "name", "")),
                    "description": str(getattr(arg, "description", "") or ""),
                    "required": bool(getattr(arg, "required", False)),
                }
                for arg in (getattr(item, "arguments", None) or [])
            ],
        }
        for item in prompts
    ]

async def get_prompt(session: ClientSession, name: str, arguments: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Render an MCP prompt and normalize its messages."""
    result = await session.get_prompt(name, arguments=arguments or {})
    messages = getattr(result, "messages", None) or []
    normalized = []
    for message in messages:
        content = getattr(message, "content", None)
        text = getattr(content, "text", None) if content is not None else None
        normalized.append({"role": str(getattr(message, "role", "user")), "text": text, "content": content})
    return normalized

async def call_tool(session: ClientSession, name: str, arguments: dict[str, Any] | None = None) -> Any:
    """Call an MCP tool and normalize structured/error responses."""
    result = await session.call_tool(name, arguments=arguments or {})
    structured = getattr(result, "structured_content", None)
    if structured is None:
        structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    content = getattr(result, "content", None) or []
    is_error = getattr(result, "is_error", None)
    if is_error is None:
        is_error = getattr(result, "isError", False)
    texts = [getattr(item, "text", str(item)) for item in content]
    return {"success": not bool(is_error), "error": "MCP tool returned an error" if is_error else None, "content": texts}

async def main() -> None:
    async with connect() as session:
        tools = await discover_tools(session)
        print("Available MCP tools:")
        for tool in tools:
            print(f"- {tool['function']['name']}: {tool['function']['description']}")
        print("\nAvailable MCP resources:")
        for resource in await discover_resources(session):
            print(f"- {resource.get('uri') or resource.get('uri_template')}: {resource['description']}")
        print("\nAvailable MCP prompts:")
        for prompt in await discover_prompts(session):
            print(f"- {prompt['name']}: {prompt['description']}")
        print("\nKolkata tool result:")
        print(await call_tool(session, "get_weather", {"location": "Kolkata"}))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
