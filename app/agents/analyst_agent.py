from __future__ import annotations

from typing import Any

from app.agents._function_calling import run_tool_calling_loop
from app.agents.base import Agent
from app.core.papers import format_known_papers
from app.infra.llm_client import GeminiClient
from app.tools.compare_papers import ComparePapersTool
from app.tools.rank_papers import RankPapersTool
from app.tools.summarize import SummarizeTool

_SYSTEM_INSTRUCTION = (
    "Você é um agente analista especializado em papers de Machine Learning. Use "
    "compare_papers para comparar papers entre si, summarize para resumir um paper "
    "específico e rank_papers para ordená-los segundo um critério. Encadeie tools "
    "quantas vezes forem necessárias e responda com um texto final baseado apenas "
    "no que as tools retornaram.\n\n"
    "A base vetorial contém exatamente estes arXiv IDs, use-os para preencher "
    "paper_id/paper_ids quando o usuário não informar os IDs explicitamente "
    "(ex: 'resuma os 5 papers', 'ranqueie todos'). Ao pedirem 'os 5 papers'/'todos', "
    "use os 5 IDs abaixo mesmo que a conversa até agora só tenha mencionado alguns "
    "deles — a base fechada tem sempre estes 5, nunca menos:\n" + format_known_papers()
)


class AnalystAgent(Agent):
    """Realiza análises comparativas, sínteses e rankings entre papers."""

    name = "analyst_agent"

    def __init__(
        self,
        llm_client: GeminiClient,
        compare_papers: ComparePapersTool,
        summarize: SummarizeTool,
        rank_papers: RankPapersTool,
    ) -> None:
        self._llm_client = llm_client
        self.tools = (compare_papers, summarize, rank_papers)

    async def handle(self, instruction: str, *, context: dict[str, Any]) -> str:
        return await run_tool_calling_loop(
            llm_client=self._llm_client,
            tools=self.tools,
            system_instruction=_SYSTEM_INSTRUCTION,
            instruction=instruction,
            history=context.get("history", []),
            agent_name=self.name,
        )
