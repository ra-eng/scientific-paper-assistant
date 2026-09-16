from __future__ import annotations

import json

from app.tools.summarize import SummarizeParams, SummarizeTool


async def test_summarize_returns_bullet_points(mock_llm_client, mock_vector_store) -> None:
    mock_vector_store.get_by_metadata.return_value = [
        {"id": "1706.03762:abstract:0", "text": "O paper propõe o Transformer, baseado em self-attention."}
    ]
    mock_llm_client.generate_structured.return_value = json.dumps(
        {"bullet_points": ["Propõe o Transformer.", "Usa self-attention.", "Elimina recorrência."]}
    )

    tool = SummarizeTool(llm_client=mock_llm_client, vector_store=mock_vector_store)
    result = await tool.run(SummarizeParams(paper_id="1706.03762", max_bullet_points=3))

    assert result.success
    assert result.data["paper_id"] == "1706.03762"
    assert len(result.data["bullet_points"]) == 3


async def test_summarize_truncates_to_max_bullet_points(mock_llm_client, mock_vector_store) -> None:
    mock_vector_store.get_by_metadata.return_value = [{"id": "1706.03762:abstract:0", "text": "texto"}]
    mock_llm_client.generate_structured.return_value = json.dumps(
        {"bullet_points": ["um", "dois", "tres", "quatro", "cinco", "seis"]}
    )

    tool = SummarizeTool(llm_client=mock_llm_client, vector_store=mock_vector_store)
    result = await tool.run(SummarizeParams(paper_id="1706.03762", max_bullet_points=2))

    assert result.success
    assert result.data["bullet_points"] == ["um", "dois"]


async def test_summarize_returns_failure_when_paper_not_found(mock_llm_client, mock_vector_store) -> None:
    mock_vector_store.get_by_metadata.return_value = []

    tool = SummarizeTool(llm_client=mock_llm_client, vector_store=mock_vector_store)
    result = await tool.run(SummarizeParams(paper_id="9999.99999"))

    assert not result.success
    assert result.error is not None
    mock_llm_client.generate_structured.assert_not_awaited()


async def test_summarize_returns_failure_on_invalid_llm_json(mock_llm_client, mock_vector_store) -> None:
    mock_vector_store.get_by_metadata.return_value = [{"id": "1706.03762:abstract:0", "text": "texto"}]
    mock_llm_client.generate_structured.return_value = "not valid json"

    tool = SummarizeTool(llm_client=mock_llm_client, vector_store=mock_vector_store)
    result = await tool.run(SummarizeParams(paper_id="1706.03762"))

    assert not result.success
    assert result.error is not None
