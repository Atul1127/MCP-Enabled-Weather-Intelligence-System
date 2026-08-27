"""Cross-encoder reranking implementation."""
from __future__ import annotations
from typing import Any
import os
from sentence_transformers import CrossEncoder

MODEL=os.environ.get("WEATHER_RERANKER_MODEL","cross-encoder/ms-marco-MiniLM-L-6-v2")
MIN_CANDIDATES=int(os.environ.get("WEATHER_RERANK_MIN_CANDIDATES","2"))
_model:CrossEncoder|None=None

def _get_model()->CrossEncoder:
    global _model
    if _model is None: _model=CrossEncoder(MODEL)
    return _model

def rerank(query:str,candidates:list[dict[str,Any]],top_k:int)->list[dict[str,Any]]:
    if not candidates:return []
    if len(candidates)<MIN_CANDIDATES:return [{**r,"reranker_score":None} for r in candidates[:top_k]]
    pairs=[(query,str(r.get("text") or r.get("narrative_text") or "")) for r in candidates]
    scores=_get_model().predict(pairs,batch_size=min(8,len(pairs)),show_progress_bar=False)
    ranked=sorted(zip(candidates,scores),key=lambda x:float(x[1]),reverse=True)
    return [{**r,"reranker_score":float(s)} for r,s in ranked[:top_k]]
