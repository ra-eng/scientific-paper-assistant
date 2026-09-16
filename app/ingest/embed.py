from __future__ import annotations

from app.infra.embeddings import EmbeddingClient
from app.infra.vector_store import VectorStoreClient
from app.ingest.chunk import Chunk


async def embed_and_upsert(
    chunks: list[Chunk],
    *,
    vector_store: VectorStoreClient,
    embedding_client: EmbeddingClient,
) -> None:
    """Gera embeddings locais para os chunks e faz upsert no ChromaDB,
    guardando paper_id/section como metadata."""
    if not chunks:
        return

    texts = [chunk.text for chunk in chunks]
    embeddings = await embedding_client.embed(texts)

    await vector_store.upsert(
        ids=[chunk.id for chunk in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"paper_id": chunk.paper_id, "section": chunk.section} for chunk in chunks],
    )
