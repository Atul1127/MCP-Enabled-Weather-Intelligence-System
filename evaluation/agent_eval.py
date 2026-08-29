"""Deterministic metrics for end-to-end agent runs."""
from __future__ import annotations

from typing import Any


def evaluate_run(result: dict[str, Any], *, expected_tools: set[str] | None = None) -> dict[str, Any]:
    observations = result.get("observations") or []
    actual_sequence = [str(item.get("tool")) for item in observations if item.get("tool")]
    actual = set(actual_sequence)
    expected = expected_tools or set()
    verification = result.get("verification") or {}
    evidence = result.get("evidence") or []
    citations = result.get("sources") or result.get("citations") or []

    duplicate_calls = len(actual_sequence) - len(set(actual_sequence))
    unexpected_calls = len(actual - expected) if expected else 0
    tool_recall = (len(actual & expected) / len(expected)) if expected else None
    tool_precision = (len(actual & expected) / len(actual)) if expected and actual else (1.0 if not expected else 0.0)
    declared_success = result.get("success", True)
    semantic_success = bool(declared_success) and bool(result.get("answer")) and not bool(result.get("errors"))
    evidence_sufficient = bool(verification.get("sufficient")) and (bool(evidence) if expected else True)

    return {
        "success": semantic_success,
        "evidence_sufficient": evidence_sufficient,
        "citation_present": bool(citations) if evidence else True,
        "tool_selection_recall": round(tool_recall, 4) if tool_recall is not None else None,
        "tool_selection_precision": round(tool_precision, 4) if expected else None,
        "tool_calls": len(actual_sequence),
        "duplicate_tool_calls": duplicate_calls,
        "unexpected_tool_calls": unexpected_calls,
        "evidence_count": len(evidence),
        "retry_count": int(result.get("retry_count", 0)),
        "verification_sufficient": bool(verification.get("sufficient")),
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"cases": 0}
    metrics = [evaluate_run(item, expected_tools=item.get("expected_tools")) for item in results]

    def rate(key: str) -> float:
        values = [m[key] for m in metrics if m[key] is not None]
        return round(sum(bool(v) for v in values) / len(values), 4) if values else 0.0

    recalls = [m["tool_selection_recall"] for m in metrics if m["tool_selection_recall"] is not None]
    precisions = [m["tool_selection_precision"] for m in metrics if m["tool_selection_precision"] is not None]
    return {
        "cases": len(results),
        "success_rate": rate("success"),
        "evidence_sufficiency_rate": rate("evidence_sufficient"),
        "citation_rate": rate("citation_present"),
        "verification_sufficiency_rate": rate("verification_sufficient"),
        "mean_tool_selection_recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "mean_tool_selection_precision": round(sum(precisions) / len(precisions), 4) if precisions else None,
        "mean_tool_calls": round(sum(m["tool_calls"] for m in metrics) / len(metrics), 3),
        "mean_duplicate_tool_calls": round(sum(m["duplicate_tool_calls"] for m in metrics) / len(metrics), 3),
        "mean_unexpected_tool_calls": round(sum(m["unexpected_tool_calls"] for m in metrics) / len(metrics), 3),
        "mean_evidence": round(sum(m["evidence_count"] for m in metrics) / len(metrics), 3),
        "recovery_rate": round(sum(m["retry_count"] > 0 for m in metrics) / len(metrics), 4),
    }
