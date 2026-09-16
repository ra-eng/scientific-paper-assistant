from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.api.dependencies import get_orchestrator, get_thread_repository
from app.infra.models import Base
from app.infra.thread_repository import ThreadRepository
from app.main import app


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    mock_orchestrator = AsyncMock()
    mock_orchestrator.ask.return_value = "resposta do orquestrador"

    async def override_get_thread_repository() -> AsyncIterator[ThreadRepository]:
        async with session_factory() as session:
            yield ThreadRepository(session)

    app.dependency_overrides[get_orchestrator] = lambda: mock_orchestrator
    app.dependency_overrides[get_thread_repository] = override_get_thread_repository

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client

    app.dependency_overrides.clear()
    await engine.dispose()


async def test_create_thread_then_list_it(client: AsyncClient) -> None:
    create_response = await client.post("/threads")
    assert create_response.status_code == 200
    thread_id = create_response.json()["thread_id"]

    list_response = await client.get("/threads")
    assert list_response.status_code == 200
    thread_ids = [thread["thread_id"] for thread in list_response.json()["threads"]]
    assert thread_id in thread_ids


async def test_post_message_persists_history_and_returns_answer(client: AsyncClient) -> None:
    thread_id = (await client.post("/threads")).json()["thread_id"]

    message_response = await client.post(f"/threads/{thread_id}/messages", json={"content": "Qual é o RAG?"})
    assert message_response.status_code == 200
    assert message_response.json()["response"] == "resposta do orquestrador"

    history_response = await client.get(f"/threads/{thread_id}/messages")
    roles = [message["role"] for message in history_response.json()["messages"]]
    assert roles == ["user", "assistant"]


async def test_post_message_to_unknown_thread_returns_404(client: AsyncClient) -> None:
    response = await client.post("/threads/does-not-exist/messages", json={"content": "oi"})
    assert response.status_code == 404


async def test_get_messages_of_unknown_thread_returns_404(client: AsyncClient) -> None:
    response = await client.get("/threads/does-not-exist/messages")
    assert response.status_code == 404


async def test_threads_do_not_share_message_history(client: AsyncClient) -> None:
    thread_a = (await client.post("/threads")).json()["thread_id"]
    thread_b = (await client.post("/threads")).json()["thread_id"]

    await client.post(f"/threads/{thread_a}/messages", json={"content": "pergunta A"})

    messages_b = await client.get(f"/threads/{thread_b}/messages")
    assert messages_b.json()["messages"] == []


async def test_post_empty_message_is_rejected(client: AsyncClient) -> None:
    thread_id = (await client.post("/threads")).json()["thread_id"]

    response = await client.post(f"/threads/{thread_id}/messages", json={"content": ""})
    assert response.status_code == 422
