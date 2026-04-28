import asyncio
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from backend.models.schemas import PromptRequest, PromptCompareRequest, StructuredOutput, CompareResult
from backend.services import llm as llm_service
from backend.services.guardrails import detect_prompt_injection
from backend.dependencies import get_redis
from backend.config import get_settings
from groq import Groq

settings = get_settings()
router = APIRouter(prefix="/prompt", tags=["prompt"])


@router.post("/run", response_model=StructuredOutput)
async def run_prompt(req: PromptRequest, redis=Depends(get_redis)):
    if detect_prompt_injection(req.prompt):
        raise HTTPException(status_code=400, detail="Prompt injection detected.")
    result = await llm_service.call_llm(
        prompt=req.prompt,
        system_prompt=req.system_prompt,
        mode=req.mode,
        output_schema=req.output_schema or None,
    )
    return result


@router.post("/compare", response_model=CompareResult)
async def compare_prompts(req: PromptCompareRequest, redis=Depends(get_redis)):
    if detect_prompt_injection(req.prompt_a) or detect_prompt_injection(req.prompt_b):
        raise HTTPException(status_code=400, detail="Prompt injection detected.")
    result_a, result_b = await asyncio.gather(
        llm_service.call_llm(req.prompt_a, req.system_prompt, output_schema=req.output_schema or None),
        llm_service.call_llm(req.prompt_b, req.system_prompt, output_schema=req.output_schema or None),
    )
    return CompareResult(result_a=result_a, result_b=result_b)


@router.post("/stream")
async def stream_prompt(req: PromptRequest):
    """
    Server-Sent Events endpoint — streams tokens as they arrive from Groq.
    Clients receive 'data: <token>\\n\\n' chunks in real time.
    """
    if detect_prompt_injection(req.prompt):
        raise HTTPException(status_code=400, detail="Prompt injection detected.")

    model = settings.smart_model if req.mode == "smart" else settings.primary_model

    def generate():
        client = Groq(api_key=settings.groq_api_key)
        messages = []
        if req.system_prompt:
            messages.append({"role": "system", "content": req.system_prompt})
        messages.append({"role": "user", "content": req.prompt})

        with client.chat.completions.stream(
            model=model,
            max_tokens=2048,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield f"data: {text}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
