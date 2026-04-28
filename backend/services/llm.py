import time
import asyncio
import json
from groq import Groq, APITimeoutError, APIConnectionError
from typing import Any
from backend.config import get_settings
from backend.models.schemas import StructuredOutput

settings = get_settings()

# Groq free-tier blended cost estimate (effectively $0, but tracked for parity)
MODEL_COSTS = {
    settings.primary_model: 0.0,
    settings.fallback_model: 0.0,
    settings.smart_model: 0.0,
}


def _get_client() -> Groq:
    return Groq(api_key=settings.groq_api_key)


def _pick_model(mode: str) -> str:
    return settings.smart_model if mode == "smart" else settings.primary_model


async def call_llm(
    prompt: str,
    system_prompt: str = "",
    mode: str = "fast",
    output_schema: dict | None = None,
    max_retries: int = 2,
) -> StructuredOutput:
    """
    Calls Groq LLM with fallback chain.
    - Primary model based on mode (fast = llama-3.1-8b, smart = llama-3.1-70b)
    - Falls back to FALLBACK_MODEL (gemma2-9b) on timeout or connection error
    - Uses JSON mode when output_schema is provided
    """
    model = _pick_model(mode)
    models_to_try = [model, settings.fallback_model]

    for attempt, current_model in enumerate(models_to_try):
        try:
            start = time.monotonic()
            client = _get_client()

            messages: list[dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            kwargs: dict[str, Any] = {
                "model": current_model,
                "max_tokens": 2048,
                "messages": messages,
            }

            if output_schema:
                # Ask the model to return valid JSON matching the schema keys
                schema_hint = json.dumps({k: f"<{type(v).__name__}>" for k, v in output_schema.items()})
                messages[-1]["content"] += f"\n\nRespond ONLY with valid JSON matching this structure: {schema_hint}"
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)
            raw_text = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0

            if output_schema:
                try:
                    data = json.loads(raw_text)
                except json.JSONDecodeError:
                    data = {"text": raw_text}
            else:
                data = {"text": raw_text}

            latency = (time.monotonic() - start) * 1000
            return StructuredOutput(
                data=data,
                raw_text=raw_text,
                tokens_used=tokens,
                latency_ms=round(latency, 2),
                model_used=current_model,
            )

        except (APITimeoutError, APIConnectionError) as e:
            if attempt == len(models_to_try) - 1:
                raise RuntimeError(f"All models failed. Last error: {e}")
            await asyncio.sleep(0.5)
            continue
        except Exception as e:
            raise RuntimeError(f"LLM call failed on {current_model}: {e}")
