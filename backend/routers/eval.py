import json
from fastapi import APIRouter, Depends
from backend.models.schemas import EvalRequest, EvalResult, DeepEvalRequest, DeepEvalResult
from backend.services.evaluation import run_ragas_eval
from backend.services.deepeval_service import run_deepeval
from backend.dependencies import get_redis

router = APIRouter(prefix="/eval", tags=["eval"])


@router.post("/run", response_model=EvalResult)
async def evaluate(req: EvalRequest, redis=Depends(get_redis)):
    return await run_ragas_eval(
        query=req.query,
        answer=req.answer,
        contexts=req.contexts,
        ground_truth=req.ground_truth,
    )


@router.post("/deepeval", response_model=DeepEvalResult)
async def deepeval_evaluate(req: DeepEvalRequest, redis=Depends(get_redis)):
    """
    DeepEval-style evaluation using LLM-as-judge via Groq.
    Scores Faithfulness, Answer Relevancy, and Hallucination.
    No ground truth required — all three metrics are reference-free.
    """
    return await run_deepeval(
        query=req.query,
        answer=req.answer,
        contexts=req.contexts,
    )


@router.get("/logs")
async def get_logs(limit: int = 100, redis=Depends(get_redis)):
    """Returns recent LLM call logs (latency, cost, model, route)."""
    if not redis:
        return {"logs": []}
    raw = await redis.lrange("llm_logs", 0, limit - 1)
    return {"logs": [json.loads(r) for r in raw]}


@router.get("/request-logs")
async def get_request_logs(limit: int = 100, redis=Depends(get_redis)):
    if not redis:
        return {"logs": []}
    raw = await redis.lrange("request_logs", 0, limit - 1)
    return {"logs": [json.loads(r) for r in raw]}


@router.get("/rag-eval-logs")
async def get_rag_eval_logs(limit: int = 100, redis=Depends(get_redis)):
    """Continuous eval scores auto-logged after every RAG chat response."""
    if not redis:
        return {"logs": []}
    raw = await redis.lrange("rag_eval_logs", 0, limit - 1)
    return {"logs": [json.loads(r) for r in raw]}
