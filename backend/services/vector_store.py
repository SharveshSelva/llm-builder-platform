from __future__ import annotations
import chromadb
import uuid
from pathlib import Path
from backend.config import get_settings
from backend.services.embeddings import embed_texts

settings = get_settings()

_client: chromadb.PersistentClient | None = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        chroma_dir = Path(settings.chroma_data_path)
        chroma_dir.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(chroma_dir))
    return _client


def _get_collection(name: str):
    client = _get_client()
    return client.get_or_create_collection(name=name)


async def add_chunks(
    texts: list[str],
    metadatas: list[dict],
    collection_name: str,
    redis_client=None,
) -> int:
    """Embed and store document chunks. Returns count added."""
    embeddings = await embed_texts(texts, redis_client)
    ids = [str(uuid.uuid4()) for _ in texts]
    collection = _get_collection(collection_name)
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    return len(texts)


async def query(
    query_text: str,
    collection_name: str,
    top_k: int = 5,
    redis_client=None,
) -> list[dict]:
    """
    Returns top_k chunks with text, metadata, and cosine distance score.
    Lower distance = more relevant.
    """
    query_embedding = await embed_texts([query_text], redis_client)
    collection = _get_collection(collection_name)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "metadata": meta,
            "score": round(1 - dist, 4),  # convert distance -> similarity
        })
    return chunks
