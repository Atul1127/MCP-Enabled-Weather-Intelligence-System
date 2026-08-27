"""Evaluate routing, planning, retrieval and citation behavior from JSONL traces.

Input records may contain:
{"intent":"knowledge","predicted_intent":"knowledge",
 "retrieved_ids":["a"],"relevant_ids":["a"],
 "citations":["S1"],"valid_citations":["S1"]}
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from evaluation.metrics import accuracy, citation_precision, mrr, ndcg_at_k, recall_at_k


def load(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(records, k=5):
    intent_pairs=[(r.get("predicted_intent"),r.get("intent")) for r in records if r.get("predicted_intent") is not None and r.get("intent") is not None]
    retrieval=[r for r in records if r.get("retrieved_ids") is not None and r.get("relevant_ids") is not None]
    return {
      "cases":len(records),
      "intent_accuracy":accuracy([x[0] for x in intent_pairs],[x[1] for x in intent_pairs]) if intent_pairs else None,
      "retrieval_recall_at_k":sum(recall_at_k(r["retrieved_ids"],r["relevant_ids"],k) for r in retrieval)/len(retrieval) if retrieval else None,
      "retrieval_mrr":sum(mrr(r["retrieved_ids"],r["relevant_ids"]) for r in retrieval)/len(retrieval) if retrieval else None,
      "retrieval_ndcg_at_k":sum(ndcg_at_k(r["retrieved_ids"],r["relevant_ids"],k) for r in retrieval)/len(retrieval) if retrieval else None,
      "citation_precision":sum(citation_precision(r.get("citations",[]),r.get("valid_citations",[])) for r in records)/len(records) if records else 0.0,
    }

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("dataset",type=Path); parser.add_argument("--k",type=int,default=5); args=parser.parse_args()
    print(json.dumps(evaluate(load(args.dataset),args.k),indent=2))
