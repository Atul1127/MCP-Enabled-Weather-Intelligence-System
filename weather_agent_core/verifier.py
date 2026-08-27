"""Deterministic evidence verifier used by the LangGraph recovery loop."""
from __future__ import annotations

from typing import Any


class EvidenceVerifier:
    """Check whether planned live/knowledge capabilities produced usable evidence."""

    def verify(
        self,
        plan: dict[str, Any],
        observations: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        errors: list[str],
    ) -> dict[str, Any]:
        successful_tools = {
            str(item.get("tool"))
            for item in observations
            if item.get("success", True) and item.get("tool")
        }
        required_groups = [
            set(step.get("preferred_tools", []))
            for step in plan.get("steps", [])
            if step.get("required", True)
        ]
        missing = [sorted(group) for group in required_groups if group and not group.intersection(successful_tools)]
        has_knowledge = any(name in successful_tools for name in {"search_weather", "ask_weather"})
        if plan.get("requires_knowledge") and not has_knowledge:
            missing.append(["search_weather", "ask_weather"])
        sufficient = not missing and bool(observations or evidence or not plan.get("requires_live_data")) and not errors
        return {
            "sufficient": sufficient,
            "successful_tools": sorted(successful_tools),
            "missing_capabilities": missing,
            "evidence_count": len(evidence),
            "errors": list(errors),
        }
