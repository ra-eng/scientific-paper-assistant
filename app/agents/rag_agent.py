from __future__ import annotations

from typing import Any

from app.agents._function_calling import run_tool_calling_loop
from app.agents.base import Agent
from app.core.papers import format_known_papers
from app.infra.llm_client import GeminiClient
from app.tools.extract_section import ExtractSectionTool
from app.tools.search_documents import SearchDocumentsTool

_SYSTEM_INSTRUCTION = (
    "Você é um agente de recuperação de informação (RAG) especializado em papers de "
    "Machine Learning. Use search_documents para busca semântica por conteúdo e "
    "extract_section quando o pedido for por uma seção específica e conhecida de um "
    "paper (ex: abstract, conclusion) — extract_section exige o paper_id exato. "
    "Encadeie tools quantas vezes forem necessárias e responda com um texto final "
    "baseado apenas no que as tools retornaram.\n\n"
    "A base vetorial contém exatamente estes arXiv IDs:\n" + format_known_papers()
)


class RAGAgent(Agent):
    """Recupera contexto relevante da base vetorial via search_documents/extract_section."""

    name = "rag_agent"

    def __init__(
        self,
        llm_client: GeminiClient,
        search_documents: SearchDocumentsTool,
        extract_section: ExtractSectionTool,
    ) -> None:
        self._llm_client = llm_client
        self.tools = (search_documents, extract_section)

    async def handle(self, instruction: str, *, context: dict[str, Any]) -> str:
        return await run_tool_calling_loop(
            llm_client=self._llm_client,
            tools=self.tools,
            system_instruction=_SYSTEM_INSTRUCTION,
            instruction=instruction,
            history=context.get("history", []),
            agent_name=self.name,
        )
