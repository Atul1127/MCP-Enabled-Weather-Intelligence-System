"""Composable local-first RAG pipeline."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from local_rag_store import get_store
from llm_provider import generate_text
from rag.query.analyzer import QueryPlan, analyze
from rag.query.expansion import expand
from rag.retrieval.dense import search as dense_search
from rag.retrieval.sparse import search as sparse_search
from rag.retrieval.hybrid import fuse
from rag.reranking.cross_encoder import rerank
from rag.context.compressor import compress
from rag.citations.validator import validate

@dataclass
class RetrievalResult:
    plan: QueryPlan
    documents: list[dict[str, Any]]
    context: str
    sources: list[dict[str, Any]]

class RAGPipeline:
    def __init__(self, *, dense_k:int=30, sparse_k:int=30, fusion_k:int=10, top_k:int=5):
        self.dense_k,self.sparse_k,self.fusion_k,self.top_k=dense_k,sparse_k,fusion_k,top_k

    def retrieve(self,query:str,*,location:str|None=None,state:str|None=None)->RetrievalResult:
        plan=analyze(query,location=location,state=state)
        store=get_store(); allowed=store.filtered_rows(location=location,state=state)
        variants=expand(plan.query,generate=generate_text) if plan.needs_expansion else [plan.query]
        dense_sets=[dense_search(store,q,self.dense_k,allowed) for q in variants]
        sparse_sets=[sparse_search(store,q,self.sparse_k,allowed) for q in variants]
        fused=fuse(dense_sets,sparse_sets,self.fusion_k)
        ranked=rerank(plan.query,fused,min(self.top_k,len(fused)))
        context,sources=compress(plan.query,ranked)
        return RetrievalResult(plan,ranked,context,sources)

    @staticmethod
    def validate_answer(answer:str,sources:list[dict[str,Any]])->tuple[str,list[dict[str,Any]]]:
        return validate(answer,sources)
