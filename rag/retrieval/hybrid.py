"""Hybrid retrieval fusion using confidence-aware reciprocal rank fusion."""
from __future__ import annotations
from typing import Any
import os

RRF_K=int(os.environ.get("WEATHER_RRF_K","60"))
CONFIDENCE_WEIGHT=float(os.environ.get("WEATHER_RRF_CONFIDENCE_WEIGHT","0.5"))

def _confidence(values:list[float], value:float)->float:
    if not values: return 0.0
    lo,hi=min(values),max(values)
    if hi<=lo: return 1.0 if value>0 else 0.0
    return max(0.0,min(1.0,(value-lo)/(hi-lo)))

def fuse(dense_results:list[dict[str,Any]], sparse_results:list[dict[str,Any]], top_k:int)->list[dict[str,Any]]:
    fused:dict[str,dict[str,Any]]={}
    for channel,results,key_name in (("dense",dense_results,"dense_score"),("bm25",sparse_results,"bm25_score")):
        values=[float(r.get(key_name,0.0)) for r in results]
        for rank,row in enumerate(results,1):
            key=str(row.get("id"))
            if not key or key=="None": continue
            confidence=_confidence(values,float(row.get(key_name,0.0)))
            item=fused.setdefault(key,{**row,"rrf_score":0.0,"confidence_score":0.0,"retrieval_ranks":[],"retrieval_channels":[]})
            item["rrf_score"]+=1/(RRF_K+rank)
            item["confidence_score"]+=CONFIDENCE_WEIGHT*confidence/(RRF_K+rank)
            item["retrieval_ranks"].append(rank)
            item["retrieval_channels"].append(channel)
            item["retrieval_confidence"]=max(float(item.get("retrieval_confidence",0.0)),confidence)
    for item in fused.values(): item["fusion_score"]=item["rrf_score"]+item["confidence_score"]
    return sorted(fused.values(),key=lambda x:x["fusion_score"],reverse=True)[:top_k]
