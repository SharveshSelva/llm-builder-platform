from fastapi import Request
from backend.config import get_settings

settings = get_settings()


async def get_redis(request: Request):
    return request.app.state.redis
