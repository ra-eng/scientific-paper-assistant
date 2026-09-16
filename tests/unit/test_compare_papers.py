from __future__ import annotations

import json

from app.tools.compare_papers import ComparePapersParams, ComparePapersTool


async def test_compare_papers_returns_structured_comparison(mock_llm_client, mock_vector_store) -> None:
    mock_vector_store.get_by_metadata.side_effect = [
        [{"id": "2210.03629:abstract:0", "text": "ReAct intercala raciocínio e ação."}],
        [{"id": "2302.04761:abstract:0", "text": "Toolformer aprende sozinho a usar ferramentas."}],
    ]
    mock_llm_client.generate_structured.return_value = json.dumps(
        {
            "comparison": "ReAct intercala pensamento e ação; Toolformer aprende via self-supervision.",
            "key_differences": ["ReAct é baseado em prompting", "Toolformer é baseado em fine-tuning"],
        }
    )

    tool = ComparePapersTool(llm_client=mock_llm_client, vector_store=mock_vector_store)
    result = await tool.run(
        ComparePapersParams(paper_ids=["2210.03629", "2302.04761"], aspect="uso de ferramentas externas")
    )

    assert result.success
    assert result.data["aspect"] == "uso de ferramentas externas"
    assert result.data["paper_ids"] == ["2210.03629", "2302.04761"]
    assert len(result.data["key_differences"]) == 2


async def test_compare_papers_returns_failure_when_a_paper_is_missing(mock_llm_client, mock_vector_store) -> None:
    mock_vector_store.get_by_metadata.side_effect = [
        [{"id": "2210.03629:abstract:0", "text": "ReAct..."}],
        [],
    ]

    tool = ComparePapersTool(llm_client=mock_llm_client, vector_store=mock_vector_store)
    result = await tool.run(ComparePapersParams(paper_ids=["2210.03629", "9999.99999"], aspect="x"))

    assert not result.success
    assert "9999.99999" in result.error
    mock_llm_client.generate_structured.assert_not_awaited()
