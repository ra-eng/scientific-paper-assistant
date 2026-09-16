from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def mock_vector_store() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_llm_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_embedding_client() -> AsyncMock:
    return AsyncMock()
