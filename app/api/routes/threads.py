from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.api.dependencies import OrchestratorDependency, ThreadRepositoryDependency
from app.api.schemas import (
    CreateThreadResponse,
    GetMessagesResponse,
    ListThreadsResponse,
    MessageItem,
    PostMessageRequest,
    PostMessageResponse,
    ThreadSummary,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/threads", tags=["threads"])


@router.post("", response_model=CreateThreadResponse)
async def create_thread(thread_repository: ThreadRepositoryDependency) -> CreateThreadResponse:
    thread = await thread_repository.create_thread()
    return CreateThreadResponse(thread_id=thread.id)


@router.get("", response_model=ListThreadsResponse)
async def list_threads(thread_repository: ThreadRepositoryDependency) -> ListThreadsResponse:
    threads = await thread_repository.list_threads()
    return ListThreadsResponse(
        threads=[ThreadSummary(thread_id=thread.id, created_at=thread.created_at) for thread in threads]
    )


@router.post("/{thread_id}/messages", response_model=PostMessageResponse)
async def post_message(
    thread_id: str,
    body: PostMessageRequest,
    thread_repository: ThreadRepositoryDependency,
    orchestrator: OrchestratorDependency,
) -> PostMessageResponse:
    thread = await thread_repository.get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Thread {thread_id} não encontrada.")

    previous_messages = await thread_repository.list_messages(thread_id)
    history = [{"role": message.role, "content": message.content} for message in previous_messages]

    await thread_repository.add_message(thread_id=thread_id, role="user", content=body.content)

    try:
        answer = await orchestrator.ask(thread_id=thread_id, question=body.content, history=history)
    except Exception as error:
        logger.exception("Falha ao processar a pergunta na thread %s", thread_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="Falha ao consultar o modelo de linguagem."
        ) from error

    await thread_repository.add_message(thread_id=thread_id, role="assistant", content=answer)

    return PostMessageResponse(thread_id=thread_id, response=answer)


@router.get("/{thread_id}/messages", response_model=GetMessagesResponse)
async def get_messages(thread_id: str, thread_repository: ThreadRepositoryDependency) -> GetMessagesResponse:
    thread = await thread_repository.get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Thread {thread_id} não encontrada.")

    messages = await thread_repository.list_messages(thread_id)
    return GetMessagesResponse(
        thread_id=thread_id,
        messages=[
            MessageItem(role=message.role, content=message.content, created_at=message.created_at)
            for message in messages
        ],
    )
