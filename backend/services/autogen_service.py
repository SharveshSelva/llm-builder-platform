"""
AutoGen-style multi-agent conversation using Groq's OpenAI-compatible endpoint.

Pattern: UserProxyAgent initiates → AssistantAgent responds → CriticAgent reviews.
This mirrors AutoGen's GroupChat pattern without the pyautogen package
(which doesn't support Python 3.14 yet), keeping the same architectural concept:
agents with distinct roles exchange messages in a structured conversation loop.
"""
import time
from groq import Groq
from backend.config import get_settings
from backend.models.schemas import AutoGenResponse

settings = get_settings()

AGENTS = {
    "assistant": (
        "You are a knowledgeable AI assistant. Answer questions clearly and accurately. "
        "When given feedback, incorporate it into an improved answer."
    ),
    "critic": (
        "You are a critical reviewer. Given a question and an AI's answer, identify: "
        "1) factual errors or unsupported claims, 2) missing key points, 3) clarity issues. "
        "Be specific and concise. End with: VERDICT: APPROVE or VERDICT: REVISE."
    ),
    "refiner": (
        "You are a response refiner. Given the original question, initial answer, and critique, "
        "produce a final improved answer that addresses all critique points."
    ),
}


def _chat(client: Groq, model: str, system: str, messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "system", "content": system}] + messages,
    )
    return response.choices[0].message.content or ""


async def run_autogen(question: str, mode: str = "fast") -> "AutoGenResponse":
    """
    3-agent AutoGen GroupChat loop:
      Round 1 — AssistantAgent answers the question
      Round 2 — CriticAgent reviews the answer
      Round 3 — If VERDICT: REVISE, RefinerAgent produces final answer
                 If VERDICT: APPROVE, initial answer is returned as final
    """
    from backend.models.schemas import AutoGenResponse

    start = time.monotonic()
    model = settings.smart_model if mode == "smart" else settings.primary_model
    client = Groq(api_key=settings.groq_api_key)

    # Round 1: Assistant answers
    initial_answer = _chat(
        client, model, AGENTS["assistant"],
        [{"role": "user", "content": question}],
    )

    # Round 2: Critic reviews
    critique = _chat(
        client, model, AGENTS["critic"],
        [
            {"role": "user", "content": f"Question: {question}\n\nAnswer to review:\n{initial_answer}"},
        ],
    )

    needs_revision = "VERDICT: REVISE" in critique.upper()

    # Round 3: Refine only if critique requested changes
    if needs_revision:
        final_answer = _chat(
            client, model, AGENTS["refiner"],
            [
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Initial answer:\n{initial_answer}\n\n"
                        f"Critique:\n{critique}\n\n"
                        f"Produce the improved final answer:"
                    ),
                }
            ],
        )
    else:
        final_answer = initial_answer

    latency = (time.monotonic() - start) * 1000
    return AutoGenResponse(
        question=question,
        initial_answer=initial_answer,
        critique=critique,
        final_answer=final_answer,
        revised=needs_revision,
        rounds=3 if needs_revision else 2,
        latency_ms=round(latency, 2),
    )
