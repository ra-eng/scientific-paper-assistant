from __future__ import annotations


class DomainError(Exception):
    """Base para erros de negócio esperados — nunca capturar como Exception genérica."""


class PaperNotFoundError(DomainError):
    """Levantado quando um paper_id referenciado não existe na base vetorial."""


class SectionNotFoundError(DomainError):
    """Levantado quando `extract_section` não encontra a seção pedida no paper."""


class ThreadNotFoundError(DomainError):
    """Levantado quando um thread_id não existe no SQLite."""


class ToolExecutionError(DomainError):
    """Levantado quando uma tool falha de forma esperada (ex: resposta do LLM inválida)."""
