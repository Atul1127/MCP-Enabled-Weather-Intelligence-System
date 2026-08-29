"""Deterministic evidence verifier used by the LangGraph recovery loop."""
from __future__ import annotations

from typing import Any


_RAG_TOOLS = {"search_weather", "ask_weather"}


class EvidenceVerifier:
    """Check whether planned required capabilities produced usable evidence."""

    @staticmethod
    def _successful_tools(observations: list[dict[str, Any]]) -> set[str]:
        return {
            str(item.get("tool"))
            for item in observations
            if item.get("tool")
            and not (
                isinstance(item.get("result"), dict)
                and item["result"].get("success") is False
            )
        }

    @staticmethod
    def _rag_evidence_usable(observations: list[dict[str, Any]]) -> bool:
        """Require retrieval payload, not merely a successful tool call."""
        for item in observations:
            if item.get("tool") not in _RAG_TOOLS:
                continue
            result = item.get("result")
            if not isinstance(result, dict) or result.get("success") is False:
                continue
            documents = result.get("documents") or []
            sources = result.get("sources") or []
            context = str(result.get("context") or "").strip()
            if documents or sources or context:
                return True
        return False

    def verify(
        self,
        plan: dict[str, Any],
        observations: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        errors: list[str],
    ) -> dict[str, Any]:
        successful_tools = self._successful_tools(observations)
        required_groups = [
            set(step.get("preferred_tools", []))
            for step in plan.get("steps", [])
            if step.get("required", True)
        ]
        missing = [
            sorted(group)
            for group in required_groups
            if group and not group.intersection(successful_tools)
        ]

        required_knowledge = any(
            step.get("required", True)
            and str(step.get("capability", "")).lower() == "knowledge"
            for step in plan.get("steps", [])
        )
        has_knowledge_tool = bool(_RAG_TOOLS.intersection(successful_tools))
        has_knowledge_evidence = self._rag_evidence_usable(observations)

        if required_knowledge and not has_knowledge_tool:
            if sorted(_RAG_TOOLS) not in missing:
                missing.append(sorted(_RAG_TOOLS))
        elif required_knowledge and not has_knowledge_evidence:
            missing.append(["usable_knowledge_evidence"])

        if required_knowledge:
            sufficient = not missing and has_knowledge_evidence
        else:
            sufficient = not missing and bool(evidence or observations or not plan.get("requires_live_data"))

        return {
            "sufficient": sufficient,
            "successful_tools": sorted(successful_tools),
            "missing_capabilities": missing,
            "evidence_count": len(evidence),
            "errors": list(errors),
        }
