import io
import time
import json
import asyncio
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.config import get_settings
from backend.services import vector_store, llm as llm_service
from backend.models.schemas import ChatResponse, Citation

settings = get_settings()


def _chunk_pdf(file_bytes: bytes, source_name: str) -> tuple[list[str], list[dict]]:
    """
    Reads PDF, splits into overlapping chunks.
    Returns (texts, metadatas).

    Chunking strategy:
    - CHUNK_SIZE tokens (default 512) keeps context meaningful
    - CHUNK_OVERLAP (default 64) prevents answer split across boundaries
    - RecursiveCharacterTextSplitter respects sentence boundaries
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    full_text_by_page = []
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        if text.strip():
            full_text_by_page.append((page_num, text))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    texts, metadatas = [], []
    for page_num, page_text in full_text_by_page:
        chunks = splitter.split_text(page_text)
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({"source": source_name, "page": page_num})

    return texts, metadatas


async def ingest_pdf(
    file_bytes: bytes,
    filename: str,
    collection_name: str,
    redis_client=None,
) -> dict:
    texts, metadatas = _chunk_pdf(file_bytes, filename)
    count = await vector_store.add_chunks(texts, metadatas, collection_name, redis_client)
    return {"filename": filename, "chunks_added": count}


async def _log_eval_async(query: str, answer: str, contexts: list[str], redis_client) -> None:
    """
    Fire-and-forget: scores the RAG response with DeepEval metrics and
    pushes results to Redis key 'rag_eval_logs' for the dashboard.
    Runs after the response is already returned to the user.
    """
    try:
        from backend.services.deepeval_service import run_deepeval
        result = await run_deepeval(query, answer, contexts)
        entry = {
            "query": query[:120],
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "hallucination": result.hallucination,
            "latency_ms": result.latency_ms,
        }
        await redis_client.lpush("rag_eval_logs", json.dumps(entry))
        await redis_client.ltrim("rag_eval_logs", 0, 999)
    except Exception:
        pass  # eval logging must never break the main response


async def chat(
    query: str,
    collection_name: str,
    top_k: int,
    mode: str,
    redis_client=None,
) -> ChatResponse:
    start = time.monotonic()

    # Retrieve
    chunks = await vector_store.query(query, collection_name, top_k, redis_client)
    if not chunks:
        return ChatResponse(
            answer="No relevant documents found. Please upload PDFs first.",
            citations=[],
            tokens_used=0,
            latency_ms=0,
            model_used="none",
        )

    # Build context
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[{i+1}] (source: {chunk['metadata']['source']}, page {chunk['metadata'].get('page','?')})\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system = (
        "You are a helpful assistant. Answer using ONLY the provided context. "
        "Cite sources by their [number] in brackets. "
        "If the answer is not in the context, say so explicitly."
    )
    prompt = f"Context:\n{context}\n\nQuestion: {query}"

    result = await llm_service.call_llm(prompt, system_prompt=system, mode=mode)

    citations = [
        Citation(
            chunk_id=str(i),
            source=c["metadata"]["source"],
            page=c["metadata"].get("page"),
            text=c["text"][:200],
            score=c["score"],
        )
        for i, c in enumerate(chunks)
    ]

    latency = (time.monotonic() - start) * 1000

    # Continuous eval: score this response in the background without blocking
    if redis_client:
        raw_contexts = [c["text"] for c in chunks]
        asyncio.create_task(_log_eval_async(query, result.raw_text, raw_contexts, redis_client))

    return ChatResponse(
        answer=result.raw_text,
        citations=citations,
        tokens_used=result.tokens_used,
        latency_ms=round(latency, 2),
        model_used=result.model_used,
    )
