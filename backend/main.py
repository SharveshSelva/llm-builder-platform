import logging
import redis.asyncio as aioredis
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import get_settings
from backend.routers import prompt, rag, agents, eval
from backend.middleware.request_logger import RequestLoggerMiddleware

settings = get_settings()
logging.basicConfig(level=settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        app.state.redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        await app.state.redis.ping()
        logging.info("Redis connected")
    except Exception as e:
        logging.warning(f"Redis unavailable, running without cache: {e}")
        app.state.redis = None
    yield
    # Shutdown
    if app.state.redis:
        await app.state.redis.close()


app = FastAPI(
    title="Production AI Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggerMiddleware)

app.include_router(prompt.router)
app.include_router(rag.router)
app.include_router(agents.router)
app.include_router(eval.router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
