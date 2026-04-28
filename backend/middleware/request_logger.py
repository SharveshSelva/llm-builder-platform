import time
import uuid
import json
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from backend.config import get_settings

settings = get_settings()
logger = logging.getLogger("request_logger")

# Groq free tier — $0 cost, tracked as 0 for dashboard parity
MODEL_COSTS = {
    settings.primary_model: 0.0,
    settings.fallback_model: 0.0,
    settings.smart_model: 0.0,
}


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, redis_client=None):
        super().__init__(app)
        self.redis = redis_client

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        start = time.monotonic()
        response = await call_next(request)
        latency = (time.monotonic() - start) * 1000

        log_entry = {
            "request_id": request_id,
            "route": str(request.url.path),
            "method": request.method,
            "latency_ms": round(latency, 2),
            "status_code": response.status_code,
        }

        if self.redis:
            await self.redis.lpush("request_logs", json.dumps(log_entry))
            await self.redis.ltrim("request_logs", 0, 999)  # keep last 1000

        logger.info(json.dumps(log_entry))
        return response


async def log_llm_call(
    redis_client,
    route: str,
    model: str,
    tokens: int,
    latency_ms: float,
    error: str | None = None,
):
    """Call this from routers after LLM responses to log cost metrics."""
    cost = round((tokens / 1_000_000) * MODEL_COSTS.get(model, 0.0), 6)
    entry = {
        "route": route,
        "model": model,
        "tokens": tokens,
        "latency_ms": latency_ms,
        "cost_usd": cost,
        "error": error,
    }
    if redis_client:
        await redis_client.lpush("llm_logs", json.dumps(entry))
        await redis_client.ltrim("llm_logs", 0, 4999)
