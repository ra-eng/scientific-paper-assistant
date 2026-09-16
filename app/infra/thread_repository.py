from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.models import Message, Thread


class ThreadRepository:
    """Acesso a threads e mensagens persistidas em SQLite. Cada thread é
    isolada: mensagens só são lidas/escritas filtradas por thread_id."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_thread(self) -> Thread:
        thread = Thread()
        self._session.add(thread)
        await self._session.commit()
        await self._session.refresh(thread)
        return thread

    async def list_threads(self) -> list[Thread]:
        result = await self._session.execute(select(Thread).order_by(Thread.created_at))
        return list(result.scalars().all())

    async def get_thread(self, thread_id: str) -> Thread | None:
        return await self._session.get(Thread, thread_id)

    async def list_messages(self, thread_id: str) -> list[Message]:
        result = await self._session.execute(
            select(Message).where(Message.thread_id == thread_id).order_by(Message.created_at)
        )
        return list(result.scalars().all())

    async def add_message(self, *, thread_id: str, role: str, content: str) -> Message:
        message = Message(thread_id=thread_id, role=role, content=content)
        self._session.add(message)
        await self._session.commit()
        await self._session.refresh(message)
        return message
