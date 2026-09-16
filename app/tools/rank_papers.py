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
    "Você ranqueia papers científicos de Machine Learning segundo um critério pedido, "
    "justificando cada posição com base apenas nos textos fornecidos."
)


class RankPapersParams(BaseModel):
    paper_ids: list[str] = Field(..., min_length=2, description="arXiv IDs a ranquear")
    criterion: str = Field(..., description="Critério de ranqueamento, ex: 'relevância para agentes com tools'")


class _RankedPaper(BaseModel):
    paper_id: str
    position: int = Field(..., ge=1, description="Posição no ranking, 1 = primeiro colocado")
    justification: str


class _RankingOutput(BaseModel):
    ranking: list[_RankedPaper]


class RankPapersTool(Tool[RankPapersParams]):
    """Ranqueia os papers segundo um critério fornecido, com justificativa para cada posição."""

    name = "rank_papers"
    description = "Ranqueia os papers segundo um critério fornecido, com justificativa para cada posição."
    params_model = RankPapersParams

    def __init__(self, llm_client: GeminiClient, vector_store: VectorStoreClient) -> None:
        self._llm_client = llm_client
        self._vector_store = vector_store

    async def run(self, params: RankPapersParams) -> ToolResult:
        try:
            context_texts = await asyncio.gather(
                *(fetch_paper_context(self._vector_store, paper_id) for paper_id in params.paper_ids)
            )
        except Exception as error:
            logger.exception("rank_papers: falha ao buscar contexto dos papers %r", params.paper_ids)
            return ToolResult(success=False, error=f"Falha ao buscar os papers na base vetorial: {error}")

        contexts = dict(zip(params.paper_ids, context_texts, strict=True))
        missing = [paper_id for paper_id, text in contexts.items() if not text]
        if missing:
            return ToolResult(success=False, error=f"Papers não encontrados na base vetorial: {', '.join(missing)}")

        context_blocks = "\n\n".join(f"--- PAPER {paper_id} ---\n{text}" for paper_id, text in contexts.items())
        prompt = (
            f"Ranqueie os papers abaixo segundo o critério: '{params.criterion}'. "
            "Toda posição do ranking precisa de uma justificativa baseada exclusivamente "
            "nos textos fornecidos.\n\n"
            f"{context_blocks}"
        )

        try:
            raw_json = await self._llm_client.generate_structured(
                prompt=prompt,
                response_schema=_RankingOutput.model_json_schema(),
                system_instruction=_SYSTEM_INSTRUCTION,
            )
            ranking = _RankingOutput.model_validate_json(raw_json)
        except Exception as error:
            logger.exception("rank_papers: falha ao gerar ranking para %r", params.paper_ids)
            return ToolResult(success=False, error=f"Falha ao gerar o ranking: {error}")

        return ToolResult(
            success=True,
            data={"criterion": params.criterion, "ranking": [item.model_dump() for item in ranking.ranking]},
        )
