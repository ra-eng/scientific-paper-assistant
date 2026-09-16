from __future__ import annotations

import asyncio

from sentence_transformers import SentenceTransformer

from app.core.settings import settings

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


class EmbeddingClient:
    """Wrapper assíncrono sobre o modelo de embedding local (sentence-transformers).
    Compartilhado entre a ingestão (embeda chunks) e o RAGAgent (embeda queries) —
    ambos precisam do mesmo modelo para os vetores caírem no mesmo espaço."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = _get_model()
        embeddings = await asyncio.to_thread(model.encode, texts)
        return [embedding.tolist() for embedding in embeddings]

    async def embed_one(self, text: str) -> list[float]:
        [embedding] = await self.embed([text])
        return embedding
