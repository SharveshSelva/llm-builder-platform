import hashlib
import json
from sentence_transformers import SentenceTransformer
from backend.config import get_settings

settings = get_settings()
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def _cache_key(text: str) -> str:
    return f"embed:{hashlib.md5(text.encode()).hexdigest()}"


async def embed_texts(texts: list[str], redis_client=None) -> list[list[float]]:
    """
    Embed a list of texts. Uses Redis cache to avoid re-embedding identical content.
    Cache key = MD5(text). TTL = EMBEDDING_CACHE_TTL seconds.
    """
    model = _get_model()
    results: list[list[float] | None] = [None] * len(texts)

    uncached_indices = []
    uncached_texts = []

    if redis_client:
        for i, text in enumerate(texts):
            key = _cache_key(text)
            cached = await redis_client.get(key)
            if cached:
                results[i] = json.loads(cached)
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
    else:
        uncached_indices = list(range(len(texts)))
        uncached_texts = texts

    if uncached_texts:
        embeddings = model.encode(uncached_texts, convert_to_numpy=True).tolist()
        for idx, emb in zip(uncached_indices, embeddings):
            results[idx] = emb
            if redis_client:
                key = _cache_key(texts[idx])
                await redis_client.setex(
                    key, settings.embedding_cache_ttl, json.dumps(emb)
                )

    return results  # type: ignore
