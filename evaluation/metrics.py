"""Model-free evaluation metrics for agent, retrieval, and grounding quality."""
from __future__ import annotations
from math import log2
from typing import Iterable, Sequence


def accuracy(predictions: Sequence[str], references: Sequence[str]) -> float:
    if not references or len(predictions) != len(references): return 0.0
    return sum(str(p).lower().strip() == str(r).lower().strip() for p, r in zip(predictions, references)) / len(references)


def precision_recall_f1(predicted: Iterable[str], relevant: Iterable[str]) -> tuple[float, float, float]:
    p, r = set(predicted), set(relevant)
    if not p and not r: return 1.0, 1.0, 1.0
    if not p or not r: return 0.0, 0.0, 0.0
    precision=len(p&r)/len(p); recall=len(p&r)/len(r)
    return precision, recall, 2*precision*recall/(precision+recall) if precision+recall else 0.0


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant=set(relevant)
    return len(set(retrieved[:k]) & relevant)/len(relevant) if relevant else 0.0


def mrr(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    relevant=set(relevant)
    for i, item in enumerate(retrieved, 1):
        if item in relevant: return 1.0/i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant=set(relevant)
    gains=[1.0 if x in relevant else 0.0 for x in retrieved[:k]]
    dcg=sum(g/log2(i+2) for i,g in enumerate(gains))
    ideal=min(len(relevant), k)
    idcg=sum(1.0/log2(i+2) for i in range(ideal))
    return dcg/idcg if idcg else 0.0


def citation_precision(citations: Iterable[str], valid_citations: Iterable[str]) -> float:
    cited=list(citations); valid=set(valid_citations)
    return len([c for c in cited if c in valid])/len(cited) if cited else 1.0


def evidence_coverage(answer_claims: Iterable[str], supported_claims: Iterable[str]) -> float:
    claims=list(answer_claims); supported=set(supported_claims)
    return len([c for c in claims if c in supported])/len(claims) if claims else 1.0
