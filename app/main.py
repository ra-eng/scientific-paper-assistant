from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.threads import router as threads_router
from app.core.logging import configure_logging
from app.infra.db import init_db

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await init_db()
    yield


app = FastAPI(title="Assistente de Análise de Artigos Científicos", lifespan=lifespan)
app.include_router(threads_router)
