from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.infra.embeddings import EmbeddingClient
from app.infra.vector_store import VectorStoreClient
from app.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)


class SearchDocumentsParams(BaseModel):
    query: str = Field(..., description="Query em linguagem natural")
    top_k: int = Field(default=5, ge=1, le=20)
    paper_id: str | None = Field(default=None, description="Restringe a busca a um paper específico")


class SearchDocumentsTool(Tool[SearchDocumentsParams]):
    """Busca semântica na base vetorial pelos chunks mais relevantes dado uma query."""

    name = "search_documents"
    description = "Busca semântica na base vetorial pelos chunks mais relevantes dado uma query."
    params_model = SearchDocumentsParams

    def __init__(self, vector_store: VectorStoreClient, embedding_client: EmbeddingClient) -> None:
        self._vector_store = vector_store
        self._embedding_client = embedding_client

    async def run(self, params: SearchDocumentsParams) -> ToolResult:
        where = {"paper_id": params.paper_id} if params.paper_id else None

        try:
            embedding = await self._embedding_client.embed_one(params.query)
            matches = await self._vector_store.query(embedding=embedding, top_k=params.top_k, where=where)
        except Exception as error:
            logger.exception("search_documents falhou para query=%r", params.query)
            return ToolResult(success=False, error=f"Falha na busca semântica: {error}")

        return ToolResult(success=True, data=matches)
