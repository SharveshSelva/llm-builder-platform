import time
from groq import Groq
from backend.config import get_settings
from backend.models.schemas import CrewResponse

settings = get_settings()


def _chat(client: Groq, model: str, system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=model,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


async def run_crew(topic: str, mode: str = "fast") -> CrewResponse:
    start = time.monotonic()
    model = settings.smart_model if mode == "smart" else settings.primary_model
    client = Groq(api_key=settings.groq_api_key)

    # Agent 1: Researcher
    researcher_notes = _chat(
        client, model,
        system="You are a Research Analyst. Produce concise, factual bullet-point notes.",
        user=f"Research this topic and list key facts, statistics, and current state:\n\n{topic}",
    )

    # Agent 2: Writer
    draft = _chat(
        client, model,
        system="You are a Content Writer. Write engaging, accessible blog posts in markdown.",
        user=(
            f"Using these research notes, write a 400-600 word blog post with "
            f"a strong intro, 3 body sections, and a conclusion.\n\nNotes:\n{researcher_notes}"
        ),
    )

    # Agent 3: Editor
    final = _chat(
        client, model,
        system="You are a Senior Editor. Polish text for clarity, flow, and correctness.",
        user=f"Edit this blog post and return the final polished version:\n\n{draft}",
    )

    latency = (time.monotonic() - start) * 1000
    return CrewResponse(
        blog_post=final,
        researcher_notes=researcher_notes,
        latency_ms=round(latency, 2),
    )
