"""Evaluate the local Weather RAG with a local Ollama judge.

Run from repository root:
    python evaluation/rag_llm_eval.py

No hosted LLM/API is required. The generator and judge both use Ollama locally.
Metrics:
- evidence/source recall against the gold topic
- citation coverage and citation validity
- LLM-judged faithfulness to retrieved evidence
- LLM-judged answer relevance to the question
- LLM-judged evidence relevance
- end-to-end latency
"""
from __future__ import annotations

import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import ollama

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advanced_rag import answer

DATASET = Path(__file__).resolve().parent / "rag_eval_dataset.json"
CORPUS = ROOT / "data" / "weather_knowledge.jsonl"
REPORT = Path(__file__).resolve().parent / "rag_llm_eval_report.json"
JUDGE_MODEL = "llama3.2:3b"


def load_corpus() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with CORPUS.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row.get("id"))] = row
    return rows


def cited_ids(text: str) -> list[str]:
    return sorted(set(re.findall(r"\[(S\d+)\]", text or "")), key=lambda x: int(x[1:]))


def topic_match(sources: list[dict[str, Any]], expected_topics: list[str]) -> bool:
    haystack = " ".join(str(s.get("topic") or "") for s in sources).lower()
    return all(any(token in haystack for token in topic.lower().split()) for topic in expected_topics)


def judge_prompt(question: str, answer_text: str, evidence: str) -> str:
    return f"""You are evaluating a grounded weather RAG answer.
Return JSON only with integer scores from 1 to 5:
{{"faithfulness":1-5,"answer_relevance":1-5,"evidence_relevance":1-5,"reason":"brief"}}

Definitions:
- faithfulness: are factual claims supported by the supplied evidence, with no invented facts?
- answer_relevance: does the answer directly and adequately answer the question?
- evidence_relevance: does the supplied evidence actually contain information useful for answering the question?
Do not reward outside knowledge. If the answer makes a claim not supported by evidence, lower faithfulness.

QUESTION:
{question}

ANSWER:
{answer_text}

RETRIEVED EVIDENCE:
{evidence}
"""


async def judge(question: str, answer_text: str, evidence: str) -> dict[str, Any]:
    try:
        response = await asyncio.to_thread(
            ollama.chat,
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": judge_prompt(question, answer_text, evidence)}],
            options={"temperature": 0},
        )
        raw = response["message"]["content"].strip()
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            raise ValueError("Judge did not return JSON")
        data = json.loads(match.group(0))
        return {
            "faithfulness": int(data.get("faithfulness", 0)),
            "answer_relevance": int(data.get("answer_relevance", 0)),
            "evidence_relevance": int(data.get("evidence_relevance", 0)),
            "reason": str(data.get("reason", "")),
        }
    except Exception as exc:
        return {"faithfulness": 0, "answer_relevance": 0, "evidence_relevance": 0, "reason": f"judge_error: {exc}"}


def build_evidence(result: dict[str, Any], corpus: dict[str, dict[str, Any]]) -> str:
    blocks: list[str] = []
    for source in result.get("sources") or []:
        sid = str(source.get("id"))
        row = corpus.get(sid)
        if not row:
            continue
        blocks.append(f"[{source.get('citation')}] Topic={row.get('topic')}; Source={row.get('source')}\n{row.get('text', '')}")
    return "\n\n---\n\n".join(blocks)


async def main() -> None:
    cases = json.loads(DATASET.read_text(encoding="utf-8"))["cases"]
    corpus = load_corpus()
    rows: list[dict[str, Any]] = []
    print(f"Running {len(cases)} local RAG evaluation cases...\n")

    for index, case in enumerate(cases, 1):
        started = time.perf_counter()
        result = await asyncio.to_thread(answer, case["question"])
        latency_ms = (time.perf_counter() - started) * 1000
        answer_text = str(result.get("answer") or "")
        sources = result.get("sources") or []
        citations = cited_ids(answer_text)
        valid_citations = [c for c in citations if any(str(s.get("citation")) == c for s in sources)]
        evidence = build_evidence(result, corpus)
        judged = await judge(case["question"], answer_text, evidence) if result.get("success") else {
            "faithfulness": 0, "answer_relevance": 0, "evidence_relevance": 0, "reason": "RAG failed"
        }
        row = {
            "id": case["id"],
            "question": case["question"],
            "success": bool(result.get("success")),
            "source_count": len(sources),
            "expected_topic_found": topic_match(sources, case.get("expected_source_topics", [])),
            "citation_count": len(citations),
            "valid_citation_count": len(valid_citations),
            "citation_coverage": round(len(valid_citations) / max(1, len(citations)), 4),
            "faithfulness": judged["faithfulness"],
            "answer_relevance": judged["answer_relevance"],
            "evidence_relevance": judged["evidence_relevance"],
            "judge_reason": judged["reason"],
            "latency_ms": round(latency_ms, 2),
            "trace_id": result.get("trace_id"),
        }
        rows.append(row)
        status = "PASS" if row["success"] and row["faithfulness"] >= 4 and row["answer_relevance"] >= 4 else "CHECK"
        print(f"[{index:02d}/{len(cases)}] {status:<5} {case['id']:<14} faith={row['faithfulness']} rel={row['answer_relevance']} evidence={row['evidence_relevance']} latency={row['latency_ms']:.0f}ms")

    successful = [r for r in rows if r["success"]]
    summary = {
        "cases": len(rows),
        "success_rate": round(statistics.mean(r["success"] for r in rows), 4),
        "evidence_topic_recall": round(statistics.mean(r["expected_topic_found"] for r in rows), 4),
        "mean_faithfulness_5": round(statistics.mean(r["faithfulness"] for r in successful), 3) if successful else 0,
        "mean_answer_relevance_5": round(statistics.mean(r["answer_relevance"] for r in successful), 3) if successful else 0,
        "mean_evidence_relevance_5": round(statistics.mean(r["evidence_relevance"] for r in successful), 3) if successful else 0,
        "citation_validity": round(statistics.mean(r["citation_coverage"] for r in successful), 4) if successful else 0,
        "mean_latency_ms": round(statistics.mean(r["latency_ms"] for r in rows), 2),
        "p50_latency_ms": round(statistics.median(r["latency_ms"] for r in rows), 2),
    }
    report = {
        "dataset": DATASET.name,
        "judge_model": JUDGE_MODEL,
        "summary": summary,
        "failures_or_checks": [r for r in rows if not r["success"] or r["faithfulness"] < 4 or r["answer_relevance"] < 4 or not r["expected_topic_found"]],
        "rows": rows,
        "notes": [
            "Generator and judge are both local Ollama calls.",
            "Faithfulness/relevance are LLM-judge metrics and should be interpreted alongside deterministic citation/evidence checks.",
            "Knowledge Graph is excluded.",
        ],
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("WEATHER MCP LOCAL RAG LLM EVALUATION")
    print("=" * 88)
    print(f"Cases: {summary['cases']}")
    print(f"Success rate           : {summary['success_rate']:.1%}")
    print(f"Evidence topic recall  : {summary['evidence_topic_recall']:.1%}")
    print(f"Faithfulness (1-5)     : {summary['mean_faithfulness_5']:.2f}")
    print(f"Answer relevance (1-5) : {summary['mean_answer_relevance_5']:.2f}")
    print(f"Evidence relevance (1-5): {summary['mean_evidence_relevance_5']:.2f}")
    print(f"Citation validity      : {summary['citation_validity']:.1%}")
    print(f"Mean latency           : {summary['mean_latency_ms']:.1f} ms")
    print(f"P50 latency            : {summary['p50_latency_ms']:.1f} ms")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    asyncio.run(main())
