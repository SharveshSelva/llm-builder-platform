from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    groq_api_key: str
    tavily_api_key: str = ""

    primary_model: str = "llama-3.1-8b-instant"
    fallback_model: str = "gemma2-9b-it"
    smart_model: str = "llama-3.1-70b-versatile"

    redis_url: str = "redis://localhost:6379"
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "documents"

    backend_url: str = "http://localhost:8000"
    log_level: str = "INFO"

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_cache_ttl: int = 86400

    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5

    chroma_data_path: str = "/tmp/chroma_data"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
