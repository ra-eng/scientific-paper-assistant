from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.core.chunk_ids import join_ordered_chunks
from app.infra.vector_store import VectorStoreClient
from app.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)


class ExtractSectionParams(BaseModel):
    paper_id: str = Field(..., description="arXiv ID do paper, ex: 1706.03762")
    section: str = Field(..., description="Nome da seção: abstract, introduction, conclusion, etc.")


class ExtractSectionTool(Tool[ExtractSectionParams]):
    """Extrai uma seção específica de um paper (ex: abstract, conclusion, introduction)."""

    name = "extract_section"
    description = "Extrai uma seção específica de um paper (ex: abstract, conclusion, introduction)."
    params_model = ExtractSectionParams

    def __init__(self, vector_store: VectorStoreClient) -> None:
        self._vector_store = vector_store

    async def run(self, params: ExtractSectionParams) -> ToolResult:
        try:
            matches = await self._vector_store.get_by_metadata(
                where={"paper_id": params.paper_id, "section": params.section}
            )
        except Exception as error:
            logger.exception("extract_section falhou para paper_id=%r section=%r", params.paper_id, params.section)
            return ToolResult(success=False, error=f"Falha ao consultar a base vetorial: {error}")

        if not matches:
            return ToolResult(
                success=False,
                error=f"Seção '{params.section}' não encontrada para o paper {params.paper_id}.",
            )

        section_text = join_ordered_chunks(matches)

        return ToolResult(
            success=True,
            data={"paper_id": params.paper_id, "section": params.section, "text": section_text},
        )
