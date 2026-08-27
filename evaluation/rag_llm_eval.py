"""Evaluate the modular Weather RAG stack with Gemini generation and judging."""
from __future__ import annotations
import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from llm_provider import generate_text, model_name
from rag.pipeline import RAGPipeline
DATASET = Path(__file__).resolve().parent / "rag_eval_dataset.json"
REPORT = Path(__file__).resolve().parent / "rag_llm_eval_report.json"
PIPELINE = RAGPipeline()


def cited_ids(text: str) -> list[str]:
    return sorted(set(re.findall(r"\[(S\d+)\]", text or "")), key=lambda x: int(x[1:]))


def topic_match(sources: list[dict[str, Any]], expected_topics: list[str]) -> bool:
    if not expected_topics:
        return True
    hay = " ".join(str(s.get("topic") or "") for s in sources).lower()
    return all(any(token in hay for token in topic.lower().split()) for topic in expected_topics)


def answer_prompt(question: str, evidence: str) -> str:
    return f"""Answer the weather knowledge question using only the supplied evidence. Do not invent facts. Cite factual claims with the supplied [S#] identifiers. If the evidence is insufficient, say so.\nQUESTION:\n{question}\nEVIDENCE:\n{evidence}"""


def judge_prompt(question: str, answer: str, evidence: str) -> str:
    return f'''You evaluate a grounded weather RAG answer. Return JSON only: {{"faithfulness":1-5,"answer_relevance":1-5,"evidence_relevance":1-5,"reason":"brief"}}. Faithfulness means claims are supported by evidence. Answer relevance means the answer directly addresses the question. Evidence relevance means the evidence is useful. Do not use outside knowledge.\nQUESTION:\n{question}\nANSWER:\n{answer}\nEVIDENCE:\n{evidence}'''

async def judge(question: str, answer: str, evidence: str) -> dict[str, Any]:
    try:
        raw = await asyncio.to_thread(generate_text, [{"role": "user", "content": judge_prompt(question, answer, evidence)}], temperature=0.0)
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            raise ValueError("judge did not return JSON")
        data = json.loads(match.group(0))
        return {"faithfulness": int(data.get("faithfulness", 0)), "answer_relevance": int(data.get("answer_relevance", 0)), "evidence_relevance": int(data.get("evidence_relevance", 0)), "reason": str(data.get("reason", ""))}
    except Exception as exc:
        return {"faithfulness": 0, "answer_relevance": 0, "evidence_relevance": 0, "reason": f"judge_error: {exc}"}

async def main() -> None:
    cases = json.loads(DATASET.read_text(encoding="utf-8"))["cases"]
    rows = []
    for case in cases:
        started = time.perf_counter()
        result = await asyncio.to_thread(PIPELINE.retrieve, case["question"])
        retrieval_latency = (time.perf_counter() - started) * 1000
        evidence = result.context
        raw_answer = await asyncio.to_thread(generate_text, [{"role": "user", "content": answer_prompt(case["question"], evidence)}], temperature=0.0)
        answer, cited_sources = PIPELINE.validate_answer(raw_answer, result.sources)
        judged = await judge(case["question"], answer, evidence)
        valid_citations = {s.get("citation") for s in result.sources if s.get("citation")}
        citations = cited_ids(answer)
        rows.append({
            "id": case["id"], "source_count": len(result.sources), "expected_topic_found": topic_match(result.sources, case.get("expected_source_topics", [])),
            "citation_count": len(citations), "valid_citation_count": len([c for c in citations if c in valid_citations]),
            "faithfulness": judged["faithfulness"], "answer_relevance": judged["answer_relevance"], "evidence_relevance": judged["evidence_relevance"],
            "judge_reason": judged["reason"], "retrieval_latency_ms": round(retrieval_latency, 2), "answer": answer, "sources": cited_sources,
        })
    summary = {
        "cases": len(rows), "evidence_topic_recall": statistics.mean(r["expected_topic_found"] for r in rows) if rows else 0,
        "mean_faithfulness_5": statistics.mean(r["faithfulness"] for r in rows) if rows else 0,
        "mean_answer_relevance_5": statistics.mean(r["answer_relevance"] for r in rows) if rows else 0,
        "mean_evidence_relevance_5": statistics.mean(r["evidence_relevance"] for r in rows) if rows else 0,
        "citation_validity": sum(r["valid_citation_count"] for r in rows) / max(1, sum(r["citation_count"] for r in rows)),
        "mean_retrieval_latency_ms": statistics.mean(r["retrieval_latency_ms"] for r in rows) if rows else 0,
        "p50_retrieval_latency_ms": statistics.median(r["retrieval_latency_ms"] for r in rows) if rows else 0,
    }
    REPORT.write_text(json.dumps({"dataset": DATASET.name, "model": model_name(), "summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
