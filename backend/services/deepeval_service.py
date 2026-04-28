"""
DeepEval evaluation service — alternative to RAGAS.

Implements the same three metrics using the LLM-as-judge pattern directly
via Groq (no DeepEval package needed, avoiding Python 3.14 conflicts).
This mirrors DeepEval's metric definitions faithfully.
"""
import json
import time
from groq import Groq
from backend.config import get_settings
from backend.models.schemas import DeepEvalResult

settings = get_settings()

FAITHFULNESS_PROMPT = """You are an expert evaluator assessing answer faithfulness.

Question: {question}
Answer: {answer}
Retrieved Contexts:
{contexts}

Task: Determine what fraction of claims in the Answer are directly supported by the Contexts.
Score 0.0 (no claims supported) to 1.0 (all claims supported).
Respond ONLY with a JSON object: {{"score": <float>, "reason": "<one sentence>"}}"""

RELEVANCY_PROMPT = """You are an expert evaluator assessing answer relevancy.

Question: {question}
Answer: {answer}

Task: How well does the Answer address the Question?
Score 0.0 (completely off-topic) to 1.0 (directly and fully answers the question).
Respond ONLY with a JSON object: {{"score": <float>, "reason": "<one sentence>"}}"""

HALLUCINATION_PROMPT = """You are an expert evaluator detecting hallucination.

Answer: {answer}
Retrieved Contexts:
{contexts}

Task: What fraction of the Answer contains information NOT present in or contradicted by the Contexts?
Score 0.0 (no hallucination) to 1.0 (completely hallucinated).
Respond ONLY with a JSON object: {{"score": <float>, "reason": "<one sentence>"}}"""


def _score(client: Groq, model: str, prompt: str) -> tuple[float, str]:
    resp = client.chat.completions.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    try:
        parsed = json.loads(resp.choices[0].message.content or "{}")
        return float(parsed.get("score", 0.0)), str(parsed.get("reason", ""))
    except (json.JSONDecodeError, ValueError):
        return 0.0, "parse error"


async def run_deepeval(
    query: str,
    answer: str,
    contexts: list[str],
) -> "DeepEvalResult":
    start = time.monotonic()
    model = settings.primary_model
    client = Groq(api_key=settings.groq_api_key)
    ctx_text = "\n---\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))

    faithfulness, faith_reason = _score(
        client, model,
        FAITHFULNESS_PROMPT.format(question=query, answer=answer, contexts=ctx_text),
    )
    relevancy, rel_reason = _score(
        client, model,
        RELEVANCY_PROMPT.format(question=query, answer=answer),
    )
    hallucination, hall_reason = _score(
        client, model,
        HALLUCINATION_PROMPT.format(answer=answer, contexts=ctx_text),
    )

    latency = (time.monotonic() - start) * 1000
    return DeepEvalResult(
        faithfulness=round(faithfulness, 4),
        answer_relevancy=round(relevancy, 4),
        hallucination=round(hallucination, 4),
        faithfulness_reason=faith_reason,
        relevancy_reason=rel_reason,
        hallucination_reason=hall_reason,
        latency_ms=round(latency, 2),
    )
