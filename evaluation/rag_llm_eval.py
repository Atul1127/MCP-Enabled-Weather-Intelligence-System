"""Evaluate the modular Weather RAG stack with a Gemini judge.

Run from repository root:
    python evaluation/rag_llm_eval.py
"""
from __future__ import annotations
import asyncio, json, re, statistics, sys, time
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from llm_provider import generate_text, model_name
from rag.pipeline import RAGPipeline
DATASET=Path(__file__).resolve().parent/"rag_eval_dataset.json"
REPORT=Path(__file__).resolve().parent/"rag_llm_eval_report.json"
PIPELINE=RAGPipeline()

def cited_ids(text:str)->list[str]: return sorted(set(re.findall(r"\[(S\d+)\]",text or "")),key=lambda x:int(x[1:]))
def topic_match(sources:list[dict[str,Any]], expected_topics:list[str])->bool:
    hay=" ".join(str(s.get("topic") or "") for s in sources).lower(); return all(any(token in hay for token in t.lower().split()) for t in expected_topics)

def judge_prompt(question:str,answer:str,evidence:str)->str:
    return f'''You evaluate a grounded weather RAG answer. Return JSON only: {{"faithfulness":1-5,"answer_relevance":1-5,"evidence_relevance":1-5,"reason":"brief"}}. Faithfulness means claims are supported by evidence. Answer relevance means the answer directly addresses the question. Evidence relevance means the evidence is useful. Do not use outside knowledge.\nQUESTION:\n{question}\nANSWER:\n{answer}\nEVIDENCE:\n{evidence}'''

async def judge(question:str,answer:str,evidence:str)->dict[str,Any]:
    try:
        raw=await asyncio.to_thread(generate_text,[{"role":"user","content":judge_prompt(question,answer,evidence)}],temperature=0.0)
        match=re.search(r"\{.*\}",raw,re.S)
        if not match: raise ValueError("judge did not return JSON")
        data=json.loads(match.group(0)); return {"faithfulness":int(data.get("faithfulness",0)),"answer_relevance":int(data.get("answer_relevance",0)),"evidence_relevance":int(data.get("evidence_relevance",0)),"reason":str(data.get("reason",""))}
    except Exception as exc: return {"faithfulness":0,"answer_relevance":0,"evidence_relevance":0,"reason":f"judge_error: {exc}"}

async def main()->None:
    cases=json.loads(DATASET.read_text(encoding="utf-8"))["cases"]; rows=[]
    for case in cases:
        started=time.perf_counter(); result=await asyncio.to_thread(PIPELINE.retrieve,case["question"]); latency=(time.perf_counter()-started)*1000
        sources=result.sources; evidence=result.context; answer=await judge(case["question"],"",evidence)
        citations=[s.get("citation") for s in sources if s.get("citation")]
        rows.append({"id":case["id"],"source_count":len(sources),"expected_topic_found":topic_match(sources,case.get("expected_source_topics",[])),"citation_count":len(citations),"faithfulness":answer["faithfulness"],"answer_relevance":answer["answer_relevance"],"evidence_relevance":answer["evidence_relevance"],"judge_reason":answer["reason"],"latency_ms":round(latency,2)})
    successful=rows
    summary={"cases":len(rows),"evidence_topic_recall":statistics.mean(r["expected_topic_found"] for r in rows) if rows else 0,"mean_faithfulness_5":statistics.mean(r["faithfulness"] for r in successful) if successful else 0,"mean_answer_relevance_5":statistics.mean(r["answer_relevance"] for r in successful) if successful else 0,"mean_evidence_relevance_5":statistics.mean(r["evidence_relevance"] for r in successful) if successful else 0,"mean_latency_ms":statistics.mean(r["latency_ms"] for r in rows) if rows else 0,"p50_latency_ms":statistics.median(r["latency_ms"] for r in rows) if rows else 0}
    REPORT.write_text(json.dumps({"dataset":DATASET.name,"model":model_name(),"summary":summary,"rows":rows},indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": asyncio.run(main())
