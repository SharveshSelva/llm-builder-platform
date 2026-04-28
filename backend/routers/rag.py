from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from backend.models.schemas import ChatRequest, ChatResponse
from backend.services import rag_service
from backend.services.guardrails import detect_prompt_injection, redact_pii
from backend.dependencies import get_redis

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    collection_name: str = Form("documents"),
    redis=Depends(get_redis),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted.")
    file_bytes = await file.read()
    result = await rag_service.ingest_pdf(file_bytes, file.filename, collection_name, redis)
    return result


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, redis=Depends(get_redis)):
    if detect_prompt_injection(req.query):
        raise HTTPException(status_code=400, detail="Prompt injection detected.")
    clean_query, pii_types = redact_pii(req.query)
    response = await rag_service.chat(
        query=clean_query,
        collection_name=req.collection_name,
        top_k=req.top_k,
        mode=req.mode,
        redis_client=redis,
    )
    return response
