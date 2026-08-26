"""Advanced local-first hybrid RAG for the weather knowledge base."""
from __future__ import annotations
import json, os, re, time
from typing import Any
from sentence_transformers import CrossEncoder
from llm_provider import generate_text, model_name, provider_name
from local_rag_store import get_store
from observability import emit, new_trace_id, span

LLM_MODEL=os.environ.get("WEATHER_LLM_MODEL","llama3.2:3b")
RERANKER_MODEL=os.environ.get("WEATHER_RERANKER_MODEL","cross-encoder/ms-marco-MiniLM-L-6-v2")
RRF_K=int(os.environ.get("WEATHER_RRF_K","60")); CONFIDENCE_WEIGHT=float(os.environ.get("WEATHER_RRF_CONFIDENCE_WEIGHT","0.5")); DEFAULT_TOP_K=int(os.environ.get("WEATHER_RAG_TOP_K","5")); DENSE_CANDIDATES=int(os.environ.get("WEATHER_VECTOR_CANDIDATES","30")); BM25_CANDIDATES=int(os.environ.get("WEATHER_BM25_CANDIDATES","30")); RERANK_CANDIDATES=int(os.environ.get("WEATHER_RERANK_CANDIDATES","5")); RERANK_MIN_CANDIDATES=int(os.environ.get("WEATHER_RERANK_MIN_CANDIDATES","2")); SIMPLE_FUSED_CANDIDATES=int(os.environ.get("WEATHER_SIMPLE_FUSED_CANDIDATES","3")); SIMPLE_RERANK_TOP_K=int(os.environ.get("WEATHER_SIMPLE_RERANK_TOP_K","2")); MAX_CONTEXT_CHARS=int(os.environ.get("WEATHER_RAG_MAX_CONTEXT_CHARS","9000"))
_reranker: CrossEncoder|None=None

def reranker()->CrossEncoder:
    global _reranker
    if _reranker is None: _reranker=CrossEncoder(RERANKER_MODEL)
    return _reranker

def _should_expand_query(query:str)->bool:
    text=query.lower().strip(); words=re.findall(r"\w+",text)
    return len(words)>10 and any(m in text for m in ("compare","difference","versus"," vs ","why","how does","what factors","relationship","associated","conditions","typical","forecast","outdoor"))

def _is_simple_rag_query(query:str)->bool:
    text=query.lower().strip()
    return len(re.findall(r"\w+",text))<=12 and not any(m in text for m in ("compare","difference","versus"," vs ","multiple","between","relationship"))

def _parse_expansion(raw:str)->list[str]:
    cleaned=re.sub(r"^```(?:json)?\s*|\s*```$","",raw.strip(),flags=re.I|re.S).strip(); candidates=[cleaned]; match=re.search(r"\{.*\}",cleaned,flags=re.S)
    if match and match.group(0)!=cleaned: candidates.append(match.group(0))
    for candidate in candidates:
        try: data=json.loads(candidate)
        except json.JSONDecodeError: continue
        variants=data.get("queries",[]) if isinstance(data,dict) else []
        if isinstance(variants,list): return [str(x).strip() for x in variants if str(x).strip()]
    return []

def expand_query(query:str,trace_id:str)->list[str]:
    if not _should_expand_query(query): emit("query_expansion.skipped",trace_id=trace_id,reason="simple_query"); return [query]
    prompt='Rewrite the weather search query into two short retrieval queries. Preserve location, date, hazard and activity terms. Return JSON only as {"queries":["...","..."]}. Query: '+query
    with span("query_expansion",trace_id=trace_id,provider=provider_name(),model=model_name()) as info:
        try:
            raw=generate_text([{"role":"user","content":prompt}],temperature=0); variants=_parse_expansion(raw)
            if not variants: info.update(ok=False,fallback=True,error="invalid query-expansion JSON"); emit("query_expansion.fallback",trace_id=trace_id,error="invalid query-expansion JSON"); return [query]
            info["variants"]=len(variants[:2]); return [query,*variants[:2]]
        except Exception as exc: info.update(ok=False,fallback=True,error=str(exc)); emit("query_expansion.fallback",trace_id=trace_id,error=str(exc)); return [query]

def _confidence(values:list[float],value:float)->float:
    if not values:return 0.0
    lo,hi=min(values),max(values)
    if hi<=lo:return 1.0 if value>0 else 0.0
    return max(0.0,min(1.0,(value-lo)/(hi-lo)))

def confidence_aware_rrf(result_sets:list[tuple[str,list[dict[str,Any]]]],top_k:int)->list[dict[str,Any]]:
    fused={}
    for channel,results in result_sets:
        key_name="dense_score" if channel=="dense" else "bm25_score"; values=[float(r.get(key_name,0.0)) for r in results]
        for rank,row in enumerate(results,1):
            key=str(row.get("id"))
            if not key or key=="None":continue
            confidence=_confidence(values,float(row.get(key_name,0.0))); item=fused.setdefault(key,{**row,"rrf_score":0.0,"confidence_score":0.0,"retrieval_ranks":[],"retrieval_channels":[]}); item["rrf_score"]+=1/(RRF_K+rank); item["confidence_score"]+=CONFIDENCE_WEIGHT*confidence/(RRF_K+rank); item["retrieval_ranks"].append(rank); item["retrieval_channels"].append(channel); item["retrieval_confidence"]=max(float(item.get("retrieval_confidence",0.0)),confidence)
    for item in fused.values():item["fusion_score"]=item["rrf_score"]+item["confidence_score"]
    return sorted(fused.values(),key=lambda x:x["fusion_score"],reverse=True)[:top_k]

def rrf_merge(result_sets:list[list[dict[str,Any]]],top_k:int)->list[dict[str,Any]]: return confidence_aware_rrf([("dense" if i%2==0 else "bm25",r) for i,r in enumerate(result_sets)],top_k)

def rerank(query:str,candidates:list[dict[str,Any]],top_k:int)->list[dict[str,Any]]:
    if not candidates:return []
    if len(candidates)<RERANK_MIN_CANDIDATES:return [{**r,"reranker_score":None} for r in candidates[:top_k]]
    pairs=[(query,str(r.get("text") or r.get("narrative_text") or "")) for r in candidates]; scores=reranker().predict(pairs,batch_size=min(8,len(pairs)),show_progress_bar=False); ranked=sorted(zip(candidates,scores),key=lambda x:float(x[1]),reverse=True)
    return [{**r,"reranker_score":float(s)} for r,s in ranked[:top_k]]

def _query_terms(query:str)->set[str]:return set(re.findall(r"[a-zA-Z0-9]+",query.lower()))

def compress_context(query:str,documents:list[dict[str,Any]],max_chars:int=MAX_CONTEXT_CHARS):
    terms=_query_terms(query); blocks=[]; sources=[]; used=0
    for i,row in enumerate(documents,1):
        text=str(row.get("text") or row.get("narrative_text") or "").strip(); sentences=[s.strip() for s in re.split(r"(?<=[.!?])\s+",text) if s.strip()]; ranked=sorted(sentences,key=lambda s:sum(1 for token in _query_terms(s) if token in terms),reverse=True); selected=" ".join(ranked[:4]) or text; block=f"[S{i}] Topic={row.get('topic')}; Source={row.get('source')}; Location={row.get('location') or 'general'}\n{selected}"
        if used+len(block)>max_chars:
            remaining=max_chars-used
            if remaining<250:break
            block=block[:remaining]
        blocks.append(block); used+=len(block); sources.append({"citation":f"S{i}","id":row.get("id"),"title":row.get("title"),"source":row.get("source"),"topic":row.get("topic"),"rrf_score":row.get("rrf_score"),"fusion_score":row.get("fusion_score"),"retrieval_confidence":row.get("retrieval_confidence"),"reranker_score":row.get("reranker_score")})
    return "\n\n---\n\n".join(blocks),sources

def _ensure_citations(answer_text:str,sources:list[dict[str,Any]])->str:
    if not sources:return answer_text.strip()
    valid=[str(s.get("citation")) for s in sources if s.get("citation")]; existing=set(re.findall(r"\[(S\d+)\]",answer_text or "")); missing=[c for c in valid if c not in existing]; text=(answer_text or "").strip()
    return text+("\n\nSources: "+", ".join(f"[{c}]" for c in missing) if missing else "")

SYSTEM_PROMPT="""You are an Indian Weather Intelligence assistant. Use ONLY the supplied retrieved evidence. Never invent retrieved facts. Cite factual claims with [S1], [S2], etc. If the evidence does not support a claim, say that it is not established by the retrieved sources. Distinguish reference guidance from live observations, forecasts, and official warnings. Keep the answer concise and useful."""

def answer(query:str,top_k:int=DEFAULT_TOP_K,location:str|None=None,state:str|None=None)->dict[str,Any]:
    trace_id=os.environ.get("WEATHER_TRACE_ID") or new_trace_id(); started=time.perf_counter()
    if not query or not query.strip():raise ValueError("Query cannot be empty")
    query=query.strip(); simple_query=_is_simple_rag_query(query); effective_fused_k=min(RERANK_CANDIDATES,SIMPLE_FUSED_CANDIDATES) if simple_query else RERANK_CANDIDATES; effective_top_k=min(top_k,SIMPLE_RERANK_TOP_K) if simple_query else top_k
    emit("rag.start",trace_id=trace_id,query=query,top_k=effective_top_k,backend="local",provider=provider_name(),model=model_name(),retrieval_mode="simple" if simple_query else "complex")
    try:
        with span("rag.store_load",trace_id=trace_id) as info:
            store=get_store(); allowed=store.filtered_rows(location=location,state=state); info.update(corpus=len(store.rows),filtered=len(allowed))
        with span("rag.query_planning",trace_id=trace_id) as info:
            variants=expand_query(query,trace_id); info.update(query_variants=len(variants),mode="simple" if simple_query else "complex")
        with span("retrieval",trace_id=trace_id) as info:
            dense_sets=[store.dense_search(q,DENSE_CANDIDATES,allowed) for q in variants]; sparse_sets=[store.bm25_search(q,BM25_CANDIDATES,allowed) for q in variants]; typed=[("dense",r) for r in dense_sets]+[("bm25",r) for r in sparse_sets]; fused=confidence_aware_rrf(typed,effective_fused_k); info.update(backend="local",corpus=len(store.rows),filtered=len(allowed),query_variants=len(variants),dense_candidates=sum(map(len,dense_sets)),bm25_candidates=sum(map(len,sparse_sets)),fused_candidates=len(fused),fusion="confidence-aware-rrf",mode="simple" if simple_query else "complex")
        with span("reranking",trace_id=trace_id) as info:
            ranked=rerank(query,fused,min(effective_top_k,len(fused))); info.update(candidates=len(fused),returned=len(ranked),model=RERANKER_MODEL if len(fused)>=RERANK_MIN_CANDIDATES else "skipped-small-candidate-set",mode="simple" if simple_query else "complex")
        with span("context_compression",trace_id=trace_id) as info:
            context,sources=compress_context(query,ranked); info.update(sources=len(sources),context_chars=len(context))
        if not context:emit("rag.no_evidence",trace_id=trace_id);return {"success":False,"query":query,"error":"No relevant evidence found in the local weather knowledge base.","trace_id":trace_id}
        with span("prompt_construction",trace_id=trace_id) as info:
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":f"Question:\n{query}\n\nRetrieved evidence:\n{context}"}]; info["prompt_chars"]=sum(len(str(m.get("content",""))) for m in messages); info["sources"]=len(sources)
        with span("generation",trace_id=trace_id,provider=provider_name(),model=model_name()) as info:
            text=generate_text(messages,temperature=0); info["answer_chars"]=len(text)
        with span("citation_processing",trace_id=trace_id) as info:
            text=_ensure_citations(text,sources); info["answer_chars"]=len(text); info["sources"]=len(sources)
        latency_ms=round((time.perf_counter()-started)*1000,2); emit("rag.end",trace_id=trace_id,latency_ms=latency_ms,source_count=len(sources),provider=provider_name(),model=model_name(),retrieval_mode="simple" if simple_query else "complex")
        return {"success":True,"query":query,"answer":text,"retrieval":{"backend":"local","strategy":"optional multi-query + dense + BM25 + confidence-aware RRF + adaptive cross-encoder + compression","mode":"simple" if simple_query else "complex","corpus":len(store.rows),"filtered":len(allowed),"query_variants":len(variants),"fused_candidates":len(fused),"returned":len(ranked)},"sources":sources,"trace_id":trace_id,"latency_ms":latency_ms,"llm_provider":provider_name(),"llm_model":model_name()}
    except Exception as exc:emit("rag.error",trace_id=trace_id,error=str(exc));return {"success":False,"query":query,"error":str(exc),"trace_id":trace_id}
