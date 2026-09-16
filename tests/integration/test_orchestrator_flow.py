from __future__ import annotations

import json

from google.genai import types

from app.agents.analyst_agent import AnalystAgent
from app.agents.orchestrator import OrchestratorAgent
from app.agents.rag_agent import RAGAgent
from app.tools.compare_papers import ComparePapersTool
from app.tools.extract_section import ExtractSectionTool
from app.tools.rank_papers import RankPapersTool
from app.tools.search_documents import SearchDocumentsTool
from app.tools.summarize import SummarizeTool


def _text_response(text: str) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]))]
    )


def _function_call_response(name: str, args: dict[str, object]) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))],
                )
            )
        ]
    )


def _build_orchestrator(mock_llm_client, mock_vector_store, mock_embedding_client) -> OrchestratorAgent:
    rag_agent = RAGAgent(
        llm_client=mock_llm_client,
        search_documents=SearchDocumentsTool(vector_store=mock_vector_store, embedding_client=mock_embedding_client),
        extract_section=ExtractSectionTool(vector_store=mock_vector_store),
    )
    analyst_agent = AnalystAgent(
        llm_client=mock_llm_client,
        compare_papers=ComparePapersTool(llm_client=mock_llm_client, vector_store=mock_vector_store),
        summarize=SummarizeTool(llm_client=mock_llm_client, vector_store=mock_vector_store),
        rank_papers=RankPapersTool(llm_client=mock_llm_client, vector_store=mock_vector_store),
    )
    return OrchestratorAgent(llm_client=mock_llm_client, rag_agent=rag_agent, analyst_agent=analyst_agent)


async def test_orchestrator_routes_to_rag_agent_and_returns_answer(
    mock_llm_client, mock_vector_store, mock_embedding_client
) -> None:
    """Cobre o fluxo completo orquestrador -> RAGAgent -> search_documents (tool)."""
    mock_embedding_client.embed_one.return_value = [0.1, 0.2]
    mock_vector_store.query.return_value = [
        {"id": "1706.03762:model architecture:0", "text": "self-attention relaciona todas as posições da sequência"}
    ]
    mock_llm_client.generate_with_tools.side_effect = [
        _function_call_response(
            "consult_rag_agent", {"instruction": "Explique o mecanismo central do paper 1706.03762"}
        ),
        _function_call_response("search_documents", {"query": "mecanismo central self-attention"}),
        _text_response("O mecanismo central é a self-attention, que relaciona todas as posições da sequência."),
    ]

    orchestrator = _build_orchestrator(mock_llm_client, mock_vector_store, mock_embedding_client)
    answer = await orchestrator.ask(
        thread_id="thread-1",
        question="Qual é o mecanismo central do paper Attention Is All You Need?",
        history=[],
    )

    assert "self-attention" in answer
    assert mock_llm_client.generate_with_tools.await_count == 3
    mock_vector_store.query.assert_awaited_once()


async def test_orchestrator_routes_to_analyst_agent_and_returns_answer(
    mock_llm_client, mock_vector_store, mock_embedding_client
) -> None:
    """Cobre o fluxo completo orquestrador -> AnalystAgent -> summarize (tool)."""
    mock_vector_store.get_by_metadata.return_value = [
        {"id": "1706.03762:abstract:0", "text": "O paper propõe o Transformer."}
    ]
    mock_llm_client.generate_structured.return_value = json.dumps(
        {"bullet_points": ["Propõe o Transformer.", "Usa self-attention."]}
    )
    mock_llm_client.generate_with_tools.side_effect = [
        _function_call_response("consult_analyst_agent", {"instruction": "Resuma o paper 1706.03762"}),
        _function_call_response("summarize", {"paper_id": "1706.03762", "max_bullet_points": 2}),
        _text_response("Resumo: o paper propõe o Transformer baseado em self-attention."),
    ]

    orchestrator = _build_orchestrator(mock_llm_client, mock_vector_store, mock_embedding_client)
    answer = await orchestrator.ask(thread_id="thread-2", question="Resuma o paper 1706.03762", history=[])

    assert "Transformer" in answer
    mock_llm_client.generate_structured.assert_awaited_once()


async def test_consolidate_merges_both_agent_outputs(mock_llm_client, mock_vector_store, mock_embedding_client) -> None:
    """Quando os dois agentes respondem, o orquestrador pede ao LLM uma síntese final."""
    mock_llm_client.generate.return_value = "resposta combinada"

    orchestrator = _build_orchestrator(mock_llm_client, mock_vector_store, mock_embedding_client)
    state = {
        "question": "pergunta",
        "history": [],
        "rag_instruction": None,
        "analyst_instruction": None,
        "rag_output": "achado do RAG",
        "analyst_output": "achado do Analyst",
        "final_answer": None,
    }

    update = await orchestrator._consolidate(state)

    assert update["final_answer"] == "resposta combinada"
    mock_llm_client.generate.assert_awaited_once()
