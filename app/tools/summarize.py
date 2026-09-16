from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.infra.llm_client import GeminiClient
from app.infra.vector_store import VectorStoreClient
from app.tools._grounding import fetch_paper_context
from app.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)

_SYSTEM_INSTRUCTION = (
    "Você resume papers científicos de Machine Learning de forma executiva e precisa, "
    "baseando-se apenas no texto fornecido — nunca em conhecimento externo."
)


class SummarizeParams(BaseModel):
    paper_id: str = Field(..., description="arXiv ID do paper a resumir")
    max_bullet_points: int = Field(default=5, ge=1, le=10)


class _SummaryOutput(BaseModel):
    bullet_points: list[str] = Field(..., description="Bullet points do resumo executivo, um por ideia central")


class SummarizeTool(Tool[SummarizeParams]):
    """Gera resumo estruturado de um paper específico."""

    name = "summarize"
    description = "Gera resumo estruturado de um paper específico."
    params_model = SummarizeParams

    def __init__(self, llm_client: GeminiClient, vector_store: VectorStoreClient) -> None:
        self._llm_client = llm_client
        self._vector_store = vector_store

    async def run(self, params: SummarizeParams) -> ToolResult:
        try:
            context_text = await fetch_paper_context(self._vector_store, params.paper_id)
        except Exception as error:
            logger.exception("summarize: falha ao buscar contexto do paper %r", params.paper_id)
            return ToolResult(success=False, error=f"Falha ao buscar o paper na base vetorial: {error}")

        if not context_text:
            return ToolResult(success=False, error=f"Paper {params.paper_id} não encontrado na base vetorial.")

        prompt = (
            f"Resuma o paper a seguir (arXiv {params.paper_id}) em no máximo "
            f"{params.max_bullet_points} bullet points, cada um trazendo uma ideia central e concisa. "
            f"Baseie-se exclusivamente no texto abaixo.\n\n--- TEXTO ---\n{context_text}"
        )

        try:
            raw_json = await self._llm_client.generate_structured(
                prompt=prompt,
                response_schema=_SummaryOutput.model_json_schema(),
                system_instruction=_SYSTEM_INSTRUCTION,
            )
            summary = _SummaryOutput.model_validate_json(raw_json)
        except Exception as error:
            logger.exception("summarize: falha ao gerar resumo para %r", params.paper_id)
            return ToolResult(success=False, error=f"Falha ao gerar o resumo: {error}")

        return ToolResult(
            success=True,
            data={
                "paper_id": params.paper_id,
                "bullet_points": summary.bullet_points[: params.max_bullet_points],
            },
        )
