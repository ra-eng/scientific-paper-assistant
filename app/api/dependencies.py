from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.analyst_agent import AnalystAgent
from app.agents.orchestrator import OrchestratorAgent
from app.agents.rag_agent import RAGAgent
from app.infra.db import get_session
from app.infra.embeddings import EmbeddingClient
from app.infra.llm_client import GeminiClient
from app.infra.thread_repository import ThreadRepository
from app.infra.vector_store import VectorStoreClient
from app.tools.compare_papers import ComparePapersTool
from app.tools.extract_section import ExtractSectionTool
from app.tools.rank_papers import RankPapersTool
from app.tools.search_documents import SearchDocumentsTool
from app.tools.summarize import SummarizeTool


@lru_cache
def get_llm_client() -> GeminiClient:
    return GeminiClient()


@lru_cache
def get_vector_store() -> VectorStoreClient:
    return VectorStoreClient()


@lru_cache
def get_embedding_client() -> EmbeddingClient:
    return EmbeddingClient()


@lru_cache
def get_orchestrator() -> OrchestratorAgent:
    """Monta o grafo orquestrador -> agentes -> tools uma única vez por
    processo (os clients por baixo ,Gemini, Chroma, embedding local ,
    mantêm conexões/modelo carregado que não devem ser recriados por request)."""
    llm_client = get_llm_client()
    vector_store = get_vector_store()
    embedding_client = get_embedding_client()

    rag_agent = RAGAgent(
        llm_client=llm_client,
        search_documents=SearchDocumentsTool(vector_store=vector_store, embedding_client=embedding_client),
        extract_section=ExtractSectionTool(vector_store=vector_store),
    )
    analyst_agent = AnalystAgent(
        llm_client=llm_client,
        compare_papers=ComparePapersTool(llm_client=llm_client, vector_store=vector_store),
        summarize=SummarizeTool(llm_client=llm_client, vector_store=vector_store),
        rank_papers=RankPapersTool(llm_client=llm_client, vector_store=vector_store),
    )
    return OrchestratorAgent(llm_client=llm_client, rag_agent=rag_agent, analyst_agent=analyst_agent)


def get_thread_repository(session: Annotated[AsyncSession, Depends(get_session)]) -> ThreadRepository:
    return ThreadRepository(session)


OrchestratorDependency = Annotated[OrchestratorAgent, Depends(get_orchestrator)]
ThreadRepositoryDependency = Annotated[ThreadRepository, Depends(get_thread_repository)]
