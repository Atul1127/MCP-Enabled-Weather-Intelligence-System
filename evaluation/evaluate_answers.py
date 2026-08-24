"""LLM-as-judge evaluation for generated weather answers."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ollama import AsyncClient
from weather_agent import run_agent

MODEL = os.environ.get("WEATHER_AGENT_MODEL", "llama3.2:3b")
JUDGE_MODEL = os.environ.get("WEATHER_EVAL_JUDGE_MODEL", MODEL)
DATASET = Path(__file__).with_name("answer_dataset.json")

JUDGE_PROMPT = """
You are a strict evaluator of a weather AI assistant.
Evaluate the candidate answer against the user's question and the criteria.
Do not reward verbosity. Penalize fabricated current weather, unsupported claims,
missing requested locations, failure to answer the question, or unsafe advice.
Return JSON only with integer scores from 1 to 5 for relevance, groundedness,
completeness, and safety, plus a short reason.
""".strip()


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


async def judge(ollama: AsyncClient, case: dict, answer: str) -> dict:
    prompt = (
        f"Question: {case['query']}\n"
        f"Criteria: {json.dumps(case['criteria'])}\n"
        f"Candidate answer: {answer}\n\n"
        "Return JSON with keys: relevance, groundedness, completeness, safety, reason."
    )
    response = await ollama.chat(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        options={"temperature": 0},
    )
    return parse_json(response.message.content or "{}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--output", default="evaluation/answer_results.json")
    args = parser.parse_args()

    cases = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    ollama = AsyncClient()
    results = []

    for case in cases:
        answer = await run_agent(case["query"])
        scores = await judge(ollama, case, answer)
        results.append({"id": case["id"], "query": case["query"], "answer": answer, "scores": scores})
        print(f"{case['id']}: {scores}")

    dimensions = ["relevance", "groundedness", "completeness", "safety"]
    averages = {
        key: round(sum(float(r["scores"].get(key, 0)) for r in results) / len(results), 2)
        for key in dimensions
    }
    pass_rate = round(
        sum(all(float(r["scores"].get(key, 0)) >= 4 for key in dimensions) for r in results)
        / len(results),
        4,
    )

    report = {
        "model": MODEL,
        "judge_model": JUDGE_MODEL,
        "dataset_size": len(results),
        "metrics": {**averages, "all_dimensions_ge_4_rate": pass_rate},
        "cases": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nAnswer Quality Evaluation")
    print("=" * 28)
    for key, value in report["metrics"].items():
        print(f"{key}: {value}")
    print(f"Results: {output}")


if __name__ == "__main__":
    asyncio.run(main())
