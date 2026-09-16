from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.tools.base import Tool


class Agent(ABC):
    """Entidade com responsabilidade própria, dona exclusiva de um conjunto de tools."""

    name: str
    tools: tuple[Tool[Any], ...]

    @abstractmethod
    async def handle(self, instruction: str, *, context: dict[str, Any]) -> str:
        """Recebe uma instrução do orquestrador, encadeia suas tools via
        function calling e retorna um texto consolidado para o orquestrador."""
        ...
