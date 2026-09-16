from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Thread(Base):
    __tablename__ = "threads"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    messages: Mapped[list[Message]] = relationship(back_populates="thread", order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    thread_id: Mapped[str] = mapped_column(ForeignKey("threads.id"))
    role: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    thread: Mapped[Thread] = relationship(back_populates="messages")
