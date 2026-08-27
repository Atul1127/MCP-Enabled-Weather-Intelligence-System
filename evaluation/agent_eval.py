"""Deterministic metrics for end-to-end agent runs."""
from __future__ import annotations
from typing import Any

def evaluate_run(result: dict[str, Any], *, expected_tools: set[str] | None = None) -> dict[str, Any]:
    observations = result.get("observations") or []
    actual = {str(item.get("tool")) for item in observations if item.get("tool")}
    verification = result.get("verification") or {}
    expected = expected_tools or set()
    tool_recall = (len(actual & expected) / len(expected)) if expected else None
    return {
        "success": bool(result.get("answer")) and not result.get("errors"),
        "grounded": bool(verification.get("sufficient")),
        "tool_selection_recall": round(tool_recall, 4) if tool_recall is not None else None,
        "tool_calls": len(observations),
        "evidence_count": len(result.get("evidence") or []),
        "retry_count": int(result.get("retry_count", 0)),
    }

def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"cases": 0}
    metrics = [evaluate_run(item, expected_tools=item.get("expected_tools")) for item in results]
    def rate(key: str) -> float:
        values = [m[key] for m in metrics if m[key] is not None]
        return round(sum(bool(v) for v in values) / len(values), 4) if values else 0.0
    recalls = [m["tool_selection_recall"] for m in metrics if m["tool_selection_recall"] is not None]
    return {
        "cases": len(results),
        "success_rate": rate("success"),
        "grounded_rate": rate("grounded"),
        "mean_tool_selection_recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "mean_tool_calls": round(sum(m["tool_calls"] for m in metrics) / len(metrics), 3),
        "mean_evidence": round(sum(m["evidence_count"] for m in metrics) / len(metrics), 3),
        "recovery_rate": round(sum(m["retry_count"] > 0 for m in metrics) / len(metrics), 4),
    }
