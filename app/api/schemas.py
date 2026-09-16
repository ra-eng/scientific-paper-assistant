from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class CreateThreadResponse(BaseModel):
    thread_id: str


class ThreadSummary(BaseModel):
    thread_id: str
    created_at: datetime


class ListThreadsResponse(BaseModel):
    threads: list[ThreadSummary]


class PostMessageRequest(BaseModel):
    content: str = Field(..., min_length=1)


class PostMessageResponse(BaseModel):
    thread_id: str
    response: str


class MessageItem(BaseModel):
    role: str
    content: str
    created_at: datetime


class GetMessagesResponse(BaseModel):
    thread_id: str
    messages: list[MessageItem]
