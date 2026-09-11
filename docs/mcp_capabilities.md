# MCP capability roadmap

The server currently exposes weather tools and the client dynamically discovers tools. The next protocol-level primitives are Resources and Prompts.

## Resources

Resources are application-controlled context. They should expose stable, read-only weather knowledge/configuration without performing model-selected actions.

Planned resources:
- `weather://capabilities` — capability metadata for the weather MCP server.
- `weather://policy` — public, non-secret tool-use policy.
- `weather://forecast/{location}` — read-only current forecast context for a location.

## Prompts

Prompts are reusable user/application-selected templates, not hidden system instructions.

Planned prompts:
- `weather_analysis` — structured weather analysis request.
- `compare_weather` — comparison request for multiple locations.
- `activity_risk` — activity-specific risk assessment request.

## Integration order

1. Add Resources and Prompts at the MCP server boundary.
2. Add client discovery/read/get helpers.
3. Add capability-registry entries for resources/prompts.
4. Add LangGraph nodes that explicitly decide when application context should be loaded.
5. Add protocol-level tests.

This separation preserves the MCP design distinction: tools are actions the model can call, while resources are context the application reads and prompts are reusable interaction templates.
