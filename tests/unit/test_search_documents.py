from __future__ import annotations

from app.tools.search_documents import SearchDocumentsParams, SearchDocumentsTool


async def test_search_documents_returns_matches_from_vector_store(mock_vector_store, mock_embedding_client) -> None:
    mock_embedding_client.embed_one.return_value = [0.1, 0.2, 0.3]
    mock_vector_store.query.return_value = [{"id": "1706.03762:abstract:0", "text": "chunk", "paper_id": "1706.03762"}]

    tool = SearchDocumentsTool(vector_store=mock_vector_store, embedding_client=mock_embedding_client)
    result = await tool.run(SearchDocumentsParams(query="self-attention"))

    assert result.success
    assert result.data == mock_vector_store.query.return_value
    mock_embedding_client.embed_one.assert_awaited_once_with("self-attention")
    mock_vector_store.query.assert_awaited_once_with(embedding=[0.1, 0.2, 0.3], top_k=5, where=None)


async def test_search_documents_filters_by_paper_id(mock_vector_store, mock_embedding_client) -> None:
    mock_embedding_client.embed_one.return_value = [0.1, 0.2, 0.3]
    mock_vector_store.query.return_value = []

    tool = SearchDocumentsTool(vector_store=mock_vector_store, embedding_client=mock_embedding_client)
    await tool.run(SearchDocumentsParams(query="attention", paper_id="1706.03762"))

    mock_vector_store.query.assert_awaited_once_with(
        embedding=[0.1, 0.2, 0.3], top_k=5, where={"paper_id": "1706.03762"}
    )


async def test_search_documents_returns_failure_on_vector_store_error(mock_vector_store, mock_embedding_client) -> None:
    mock_embedding_client.embed_one.return_value = [0.1]
    mock_vector_store.query.side_effect = RuntimeError("chroma unreachable")

    tool = SearchDocumentsTool(vector_store=mock_vector_store, embedding_client=mock_embedding_client)
    result = await tool.run(SearchDocumentsParams(query="attention"))

    assert not result.success
    assert result.error is not None
