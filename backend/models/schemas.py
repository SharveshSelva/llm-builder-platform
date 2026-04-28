from pydantic import BaseModel, Field
from typing import Any, Optional, Literal
from datetime import datetime


# -- Prompt Playground --------------------------------------------------------
class PromptRequest(BaseModel):
    prompt: str
    system_prompt: str = ""
    model: str = "llama-3.1-8b-instant"
    output_schema: dict[str, Any] = Field(default_factory=dict)
    mode: Literal["fast", "smart"] = "fast"


class PromptCompareRequest(BaseModel):
    prompt_a: str
    prompt_b: str
    system_prompt: str = ""
    model: str = "llama-3.1-8b-instant"
    output_schema: dict[str, Any] = Field(default_factory=dict)


class StructuredOutput(BaseModel):
    data: dict[str, Any]
    raw_text: str
    tokens_used: int
    latency_ms: float
    model_used: str


class CompareResult(BaseModel):
    result_a: StructuredOutput
    result_b: StructuredOutput


# -- RAG ----------------------------------------------------------------------
class IngestRequest(BaseModel):
    collection_name: str = "documents"


class ChatRequest(BaseModel):
    query: str
    collection_name: str = "documents"
    top_k: int = 5
    mode: Literal["fast", "smart"] = "fast"


class Citation(BaseModel):
    chunk_id: str
    source: str
    page: Optional[int]
    text: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    tokens_used: int
    latency_ms: float
    model_used: str


# -- Agents -------------------------------------------------------------------
class ResearchRequest(BaseModel):
    question: str
    max_results: int = 3
    mode: Literal["fast", "smart"] = "fast"


class ResearchResponse(BaseModel):
    summary: str
    sources: list[str]
    reflection_used: bool
    latency_ms: float


class CrewRequest(BaseModel):
    topic: str
    mode: Literal["fast", "smart"] = "fast"


class CrewResponse(BaseModel):
    blog_post: str
    researcher_notes: str
    latency_ms: float


class AutoGenRequest(BaseModel):
    question: str
    mode: Literal["fast", "smart"] = "fast"


class AutoGenResponse(BaseModel):
    question: str
    initial_answer: str
    critique: str
    final_answer: str
    revised: bool
    rounds: int
    latency_ms: float


# -- Evaluation ---------------------------------------------------------------
class EvalRequest(BaseModel):
    query: str
    answer: str
    contexts: list[str]
    ground_truth: Optional[str] = None


class EvalResult(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: Optional[float]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DeepEvalRequest(BaseModel):
    query: str
    answer: str
    contexts: list[str]


class DeepEvalResult(BaseModel):
    faithfulness: float
    answer_relevancy: float
    hallucination: float
    faithfulness_reason: str
    relevancy_reason: str
    hallucination_reason: str
    latency_ms: float


# -- Request Log --------------------------------------------------------------
class RequestLog(BaseModel):
    request_id: str
    route: str
    model_used: str
    tokens_used: int
    latency_ms: float
    cost_usd: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    error: Optional[str] = None
