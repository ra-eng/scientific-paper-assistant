from __future__ import annotations

from app.tools.extract_section import ExtractSectionParams, ExtractSectionTool


async def test_extract_section_joins_ordered_chunks(mock_vector_store) -> None:
    mock_vector_store.get_by_metadata.return_value = [
        {"id": "1706.03762:abstract:1", "text": "segunda parte."},
        {"id": "1706.03762:abstract:0", "text": "primeira parte."},
    ]

    tool = ExtractSectionTool(vector_store=mock_vector_store)
    result = await tool.run(ExtractSectionParams(paper_id="1706.03762", section="abstract"))

    assert result.success
    assert result.data["text"] == "primeira parte.\n\nsegunda parte."
    mock_vector_store.get_by_metadata.assert_awaited_once_with(where={"paper_id": "1706.03762", "section": "abstract"})


async def test_extract_section_returns_failure_when_not_found(mock_vector_store) -> None:
    mock_vector_store.get_by_metadata.return_value = []

    tool = ExtractSectionTool(vector_store=mock_vector_store)
    result = await tool.run(ExtractSectionParams(paper_id="1706.03762", section="nonexistent"))

    assert not result.success
    assert result.error is not None


async def test_extract_section_returns_failure_on_vector_store_error(mock_vector_store) -> None:
    mock_vector_store.get_by_metadata.side_effect = RuntimeError("chroma unreachable")

    tool = ExtractSectionTool(vector_store=mock_vector_store)
    result = await tool.run(ExtractSectionParams(paper_id="1706.03762", section="abstract"))

    assert not result.success
    assert result.error is not None
