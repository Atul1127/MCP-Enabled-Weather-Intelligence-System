"""Gemini-as-judge evaluation for generated weather answers."""
from __future__ import annotations
import argparse
import asyncio
import json
import re
from pathlib import Path
from llm_provider import generate_text, model_name
from agent import run_agent

DATASET = Path(__file__).with_name("answer_dataset.json")
JUDGE_PROMPT = """You are a strict evaluator of a weather AI assistant. Evaluate the candidate answer only against the user question and supplied evidence. Penalize fabricated current weather, unsupported claims, missing requested locations, failure to answer the question, or unsafe advice. Return JSON only with integer scores 1-5 for relevance, groundedness, completeness, and safety, plus a short reason."""

def parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise ValueError("Judge did not return JSON")
    return json.loads(match.group(0))

async def judge(case: dict, answer: str, evidence: list[dict]) -> dict:
    prompt = f"Question: {case['query']}\nCriteria: {json.dumps(case.get('criteria', {}))}\nCandidate answer: {answer}\nEvidence: {json.dumps(evidence, default=str)}"
    raw = await asyncio.to_thread(generate_text, [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": prompt}], temperature=0.0)
    return parse_json(raw)

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--output", default="evaluation/answer_results.json")
    args = parser.parse_args()
    cases = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    results = []
    for case in cases:
        result = await run_agent(case["query"])
        scores = await judge(case, str(result.get("answer") or ""), result.get("evidence") or [])
        results.append({"id": case["id"], "query": case["query"], "answer": result.get("answer"), "scores": scores, "trace_id": result.get("trace_id")})
        print(f"{case['id']}: {scores}")
    dimensions = ["relevance", "groundedness", "completeness", "safety"]
    averages = {key: round(sum(float(r["scores"].get(key, 0)) for r in results) / len(results), 2) for key in dimensions} if results else {key: 0.0 for key in dimensions}
    output = {"model": model_name(), "cases": len(results), "averages": averages, "results": results}
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(averages, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
