from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, Field

from app.infra.llm_client import GeminiClient
from app.infra.vector_store import VectorStoreClient
from app.tools._grounding import fetch_paper_context
from app.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)

_SYSTEM_INSTRUCTION = (
    "Você compara papers científicos de Machine Learning segundo um aspecto pedido, "
    "baseando-se apenas nos textos fornecidos — nunca em conhecimento externo."
)


class ComparePapersParams(BaseModel):
    paper_ids: list[str] = Field(..., min_length=2, description="arXiv IDs a comparar")
    aspect: str = Field(..., description="Aspecto da comparação, ex: 'uso de ferramentas externas'")


class _ComparisonOutput(BaseModel):
    comparison: str = Field(..., description="Texto comparando os papers segundo o aspecto pedido")
    key_differences: list[str] = Field(
        default_factory=list, description="Lista curta das principais diferenças encontradas"
    )


class ComparePapersTool(Tool[ComparePapersParams]):
    """Recebe lista de paper IDs e um aspecto, retorna comparação estruturada."""

    name = "compare_papers"
    description = "Recebe lista de paper IDs e um aspecto, retorna comparação estruturada."
    params_model = ComparePapersParams

    def __init__(self, llm_client: GeminiClient, vector_store: VectorStoreClient) -> None:
        self._llm_client = llm_client
        self._vector_store = vector_store

    async def run(self, params: ComparePapersParams) -> ToolResult:
        try:
            context_texts = await asyncio.gather(
                *(fetch_paper_context(self._vector_store, paper_id) for paper_id in params.paper_ids)
            )
        except Exception as error:
            logger.exception("compare_papers: falha ao buscar contexto dos papers %r", params.paper_ids)
            return ToolResult(success=False, error=f"Falha ao buscar os papers na base vetorial: {error}")

        contexts = dict(zip(params.paper_ids, context_texts, strict=True))
        missing = [paper_id for paper_id, text in contexts.items() if not text]
        if missing:
            return ToolResult(success=False, error=f"Papers não encontrados na base vetorial: {', '.join(missing)}")

        context_blocks = "\n\n".join(f"--- PAPER {paper_id} ---\n{text}" for paper_id, text in contexts.items())
        prompt = (
            f"Compare os papers abaixo quanto ao aspecto: '{params.aspect}'. "
            "Baseie-se exclusivamente nos textos fornecidos.\n\n"
            f"{context_blocks}"
        )

        try:
            raw_json = await self._llm_client.generate_structured(
                prompt=prompt,
                response_schema=_ComparisonOutput.model_json_schema(),
                system_instruction=_SYSTEM_INSTRUCTION,
            )
            comparison = _ComparisonOutput.model_validate_json(raw_json)
        except Exception as error:
            logger.exception("compare_papers: falha ao gerar comparação para %r", params.paper_ids)
            return ToolResult(success=False, error=f"Falha ao gerar a comparação: {error}")

        return ToolResult(
            success=True,
            data={"aspect": params.aspect, "paper_ids": params.paper_ids, **comparison.model_dump()},
        )
