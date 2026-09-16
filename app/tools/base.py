from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

ParamsT = TypeVar("ParamsT", bound=BaseModel)


class ToolResult(BaseModel):
    """Envelope padronizado que toda tool retorna, sucesso ou falha."""

    success: bool
    data: Any | None = None
    error: str | None = None


class Tool(ABC, Generic[ParamsT]):
    """Unidade de capacidade atômica e reutilizável. Sem memória, sem decisões."""

    name: str
    description: str
    params_model: type[ParamsT]

    @abstractmethod
    async def run(self, params: ParamsT) -> ToolResult: ...
