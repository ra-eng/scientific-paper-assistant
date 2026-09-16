from __future__ import annotations

from google.genai import types

from app.agents.rag_agent import RAGAgent
from app.tools.extract_section import ExtractSectionTool
from app.tools.search_documents import SearchDocumentsTool


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


def _build_agent(mock_llm_client, mock_vector_store, mock_embedding_client) -> RAGAgent:
    return RAGAgent(
        llm_client=mock_llm_client,
        search_documents=SearchDocumentsTool(vector_store=mock_vector_store, embedding_client=mock_embedding_client),
        extract_section=ExtractSectionTool(vector_store=mock_vector_store),
    )


async def test_handle_returns_text_when_no_tool_call_is_made(
    mock_llm_client, mock_vector_store, mock_embedding_client
) -> None:
    mock_llm_client.generate_with_tools.return_value = _text_response("resposta direta")

    agent = _build_agent(mock_llm_client, mock_vector_store, mock_embedding_client)
    result = await agent.handle("O que é atenção?", context={})

    assert result == "resposta direta"
    mock_llm_client.generate_with_tools.assert_awaited_once()


async def test_handle_executes_tool_call_and_returns_final_text(
    mock_llm_client, mock_vector_store, mock_embedding_client
) -> None:
    mock_embedding_client.embed_one.return_value = [0.1, 0.2]
    mock_vector_store.query.return_value = [{"id": "1706.03762:abstract:0", "text": "self-attention..."}]
    mock_llm_client.generate_with_tools.side_effect = [
        _function_call_response("search_documents", {"query": "self-attention"}),
        _text_response("resposta final com base na busca"),
    ]

    agent = _build_agent(mock_llm_client, mock_vector_store, mock_embedding_client)
    result = await agent.handle("Explique self-attention", context={})

    assert result == "resposta final com base na busca"
    assert mock_llm_client.generate_with_tools.await_count == 2
    mock_vector_store.query.assert_awaited_once()


async def test_handle_recovers_from_unknown_tool_call(
    mock_llm_client, mock_vector_store, mock_embedding_client
) -> None:
    mock_llm_client.generate_with_tools.side_effect = [
        _function_call_response("tool_que_nao_existe", {}),
        _text_response("segue mesmo assim"),
    ]

    agent = _build_agent(mock_llm_client, mock_vector_store, mock_embedding_client)
    result = await agent.handle("pergunta qualquer", context={})

    assert result == "segue mesmo assim"


async def test_handle_stops_after_max_rounds_without_final_text(
    mock_llm_client, mock_vector_store, mock_embedding_client
) -> None:
    mock_embedding_client.embed_one.return_value = [0.1]
    mock_vector_store.query.return_value = []
    mock_llm_client.generate_with_tools.return_value = _function_call_response(
        "search_documents", {"query": "loop infinito"}
    )

    agent = _build_agent(mock_llm_client, mock_vector_store, mock_embedding_client)
    result = await agent.handle("pergunta qualquer", context={})

    assert result == ""
    assert mock_llm_client.generate_with_tools.await_count == 4
