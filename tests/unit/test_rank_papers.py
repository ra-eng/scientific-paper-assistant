from __future__ import annotations

import json

from app.tools.rank_papers import RankPapersParams, RankPapersTool


async def test_rank_papers_returns_structured_ranking(mock_llm_client, mock_vector_store) -> None:
    mock_vector_store.get_by_metadata.side_effect = [
        [{"id": "2210.03629:abstract:0", "text": "ReAct..."}],
        [{"id": "2302.04761:abstract:0", "text": "Toolformer..."}],
    ]
    mock_llm_client.generate_structured.return_value = json.dumps(
        {
            "ranking": [
                {"paper_id": "2210.03629", "position": 1, "justification": "Mais direto para agentes com tools."},
                {"paper_id": "2302.04761", "position": 2, "justification": "Focado em fine-tuning, menos flexível."},
            ]
        }
    )

    tool = RankPapersTool(llm_client=mock_llm_client, vector_store=mock_vector_store)
    result = await tool.run(
        RankPapersParams(paper_ids=["2210.03629", "2302.04761"], criterion="relevância para agentes com tools")
    )

    assert result.success
    assert result.data["ranking"][0]["paper_id"] == "2210.03629"
    assert result.data["ranking"][0]["position"] == 1


async def test_rank_papers_returns_failure_when_a_paper_is_missing(mock_llm_client, mock_vector_store) -> None:
    mock_vector_store.get_by_metadata.side_effect = [[], [{"id": "2302.04761:abstract:0", "text": "x"}]]

    tool = RankPapersTool(llm_client=mock_llm_client, vector_store=mock_vector_store)
    result = await tool.run(RankPapersParams(paper_ids=["9999.99999", "2302.04761"], criterion="x"))

    assert not result.success
    mock_llm_client.generate_structured.assert_not_awaited()
