from __future__ import annotations

import json

from google.genai import types

from app.agents.analyst_agent import AnalystAgent
from app.tools.compare_papers import ComparePapersTool
from app.tools.rank_papers import RankPapersTool
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


def _build_agent(mock_llm_client, mock_vector_store) -> AnalystAgent:
    return AnalystAgent(
        llm_client=mock_llm_client,
        compare_papers=ComparePapersTool(llm_client=mock_llm_client, vector_store=mock_vector_store),
        summarize=SummarizeTool(llm_client=mock_llm_client, vector_store=mock_vector_store),
        rank_papers=RankPapersTool(llm_client=mock_llm_client, vector_store=mock_vector_store),
    )


async def test_handle_returns_text_when_no_tool_call_is_made(mock_llm_client, mock_vector_store) -> None:
    mock_llm_client.generate_with_tools.return_value = _text_response("resposta direta")

    agent = _build_agent(mock_llm_client, mock_vector_store)
    result = await agent.handle("Qual paper é mais relevante?", context={})

    assert result == "resposta direta"


async def test_handle_executes_summarize_tool_and_returns_final_text(mock_llm_client, mock_vector_store) -> None:
    mock_vector_store.get_by_metadata.return_value = [{"id": "1706.03762:abstract:0", "text": "texto do paper"}]
    mock_llm_client.generate_structured.return_value = json.dumps({"bullet_points": ["ponto 1", "ponto 2"]})
    mock_llm_client.generate_with_tools.side_effect = [
        _function_call_response("summarize", {"paper_id": "1706.03762"}),
        _text_response("resumo consolidado"),
    ]

    agent = _build_agent(mock_llm_client, mock_vector_store)
    result = await agent.handle("Resuma o paper 1706.03762", context={})

    assert result == "resumo consolidado"
    assert mock_llm_client.generate_with_tools.await_count == 2
