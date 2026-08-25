"""Evaluate the real advanced RAG path with a local LLM judge.

The evaluator measures citation presence, retrieval availability, latency, and
LLM-judged faithfulness/relevance. It uses the repository's existing answer
benchmark and Ollama, so no paid API is required.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ollama

import advanced_rag

MODEL = os.environ.get("WEATHER_EVAL_MODEL", os.environ.get("WEATHER_LLM_MODEL", "llama3.2:3b"))
DATASET = Path(__file__).with_name("answer_dataset.json")

JUDGE_PROMPT = """You are a strict RAG evaluator. Score the answer only against the
provided evidence and question. Return JSON only:
{"faithfulness":0-1,"relevance":0-1,"citation_quality":0-1,"notes":"short"}
Faithfulness means claims are supported by evidence. Relevance means the answer
addresses the question. Citation quality means factual claims have usable [S#]
citations. Do not reward confident unsupported claims."""


def judge(question: str, answer: str, evidence: list[dict]) -> dict:
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": json.dumps({"question": question, "answer": answer, "evidence": evidence}, ensure_ascii=False)},
        ],
        options={"temperature": 0},
    )
    text = response["message"]["content"].strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {"faithfulness": 0.0, "relevance": 0.0, "citation_quality": 0.0, "notes": "judge returned non-JSON"}
    try:
        value = json.loads(match.group(0))
        return {
            "faithfulness": float(value.get("faithfulness", 0)),
            "relevance": float(value.get("relevance", 0)),
            "citation_quality": float(value.get("citation_quality", 0)),
            "notes": str(value.get("notes", "")),
        }
    except (ValueError, TypeError, json.JSONDecodeError):
        return {"faithfulness": 0.0, "relevance": 0.0, "citation_quality": 0.0, "notes": "invalid judge output"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--output", default="evaluation/rag_results.json")
    args = parser.parse_args()
    cases = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    results = []

    for case in cases:
        started = time.perf_counter()
        try:
            result = advanced_rag.answer(case["query"], top_k=5)
            latency = round((time.perf_counter() - started) * 1000, 2)
            if not result.get("success"):
                results.append({"id": case["id"], "query": case["query"], "success": False, "latency_ms": latency, "error": result.get("error")})
                continue
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            scores = judge(case["query"], answer, sources)
            results.append({
                "id": case["id"],
                "query": case["query"],
                "success": True,
                "latency_ms": latency,
                "citation_count": len(re.findall(r"\[S\d+\]", answer)),
                "source_count": len(sources),
                **scores,
                "answer": answer,
                "trace_id": result.get("trace_id"),
            })
        except Exception as exc:
            results.append({"id": case["id"], "query": case["query"], "success": False, "error": str(exc)})

    successful = [r for r in results if r["success"]]
    report = {
        "model": MODEL,
        "dataset_size": len(results),
        "success_rate": round(len(successful) / len(results), 4) if results else 0,
        "metrics": {
            "faithfulness": round(mean(r["faithfulness"] for r in successful), 4) if successful else 0,
            "relevance": round(mean(r["relevance"] for r in successful), 4) if successful else 0,
            "citation_quality": round(mean(r["citation_quality"] for r in successful), 4) if successful else 0,
            "citation_presence_rate": round(mean(r["citation_count"] > 0 for r in successful), 4) if successful else 0,
            "average_latency_ms": round(mean(r["latency_ms"] for r in successful), 2) if successful else 0,
        },
        "cases": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
