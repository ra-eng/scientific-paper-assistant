from __future__ import annotations

import asyncio
from typing import Any

import chromadb

from app.core.settings import settings


class VectorStoreClient:
    """Wrapper fino sobre o cliente ChromaDB (modo HTTP, container próprio)."""

    def __init__(self) -> None:
        self._client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        self._collection = self._client.get_or_create_collection("papers")

    async def query(
        self,
        *,
        embedding: list[float],
        top_k: int,
        where: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Retorna os `top_k` chunks mais próximos, opcionalmente filtrados
        por metadata (ex: {"paper_id": "1706.03762", "section": "conclusion"})."""
        result = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[embedding],  # type: ignore[arg-type]
            n_results=top_k,
            where=where,  # type: ignore[arg-type]
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        return [
            {"id": id_, "text": document, "distance": distance, **(metadata or {})}
            for id_, document, metadata, distance in zip(ids, documents, metadatas, distances, strict=True)
        ]

    async def get_by_metadata(self, *, where: dict[str, str]) -> list[dict[str, Any]]:
        """Busca direta por metadata (paper_id/section), sem embedding ,
        usado por extract_section, que já sabe exatamente o que quer."""
        result = await asyncio.to_thread(self._collection.get, where=where)  # type: ignore[arg-type]
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        return [
            {"id": id_, "text": document, **(metadata or {})}
            for id_, document, metadata in zip(ids, documents, metadatas, strict=True)
        ]

    async def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, str]],
    ) -> None:
        await asyncio.to_thread(
            self._collection.upsert,
            ids=ids,
            embeddings=embeddings,  # type: ignore[arg-type]
            documents=documents,
            metadatas=metadatas,  # type: ignore[arg-type]
        )
