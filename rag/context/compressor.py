"""Deterministic query-aware context compression."""
from __future__ import annotations
from typing import Any
import os, re

MAX_CONTEXT_CHARS=int(os.environ.get("WEATHER_RAG_MAX_CONTEXT_CHARS","9000"))

def _terms(text:str)->set[str]: return set(re.findall(r"[a-zA-Z0-9]+",text.lower()))

def compress(query:str,documents:list[dict[str,Any]],max_chars:int|None=None):
    limit=max_chars or MAX_CONTEXT_CHARS; terms=_terms(query); blocks=[]; sources=[]; used=0
    for i,row in enumerate(documents,1):
        text=str(row.get("text") or row.get("narrative_text") or "").strip()
        sentences=[s.strip() for s in re.split(r"(?<=[.!?])\s+",text) if s.strip()]
        selected=sorted(sentences,key=lambda s:sum(1 for token in _terms(s) if token in terms),reverse=True)[:4]
        excerpt=" ".join(selected) or text
        block=f"[S{i}] Topic={row.get('topic')}; Source={row.get('source')}; Location={row.get('location') or 'general'}\n{excerpt}"
        if used+len(block)>limit:
            remaining=limit-used
            if remaining<250: break
            block=block[:remaining]
        blocks.append(block); used+=len(block)
        sources.append({"citation":f"S{i}","id":row.get("id"),"title":row.get("title"),"source":row.get("source"),"topic":row.get("topic"),"rrf_score":row.get("rrf_score"),"fusion_score":row.get("fusion_score"),"retrieval_confidence":row.get("retrieval_confidence"),"reranker_score":row.get("reranker_score")})
    return "\n\n---\n\n".join(blocks),sources
