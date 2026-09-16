from __future__ import annotations

from app.core.chunk_ids import join_ordered_chunks
from app.infra.vector_store import VectorStoreClient

_DEFAULT_MAX_CHARS = 6000


async def fetch_paper_context(
    vector_store: VectorStoreClient,
    paper_id: str,
    *,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> str:
    """Reconstrói um trecho representativo do paper (todos os chunks, na
    ordem original) para servir de grounding para as tools do AnalystAgent.
    Trunca em max_chars para não estourar o prompt em papers longos.
    Retorna string vazia se o paper_id não existir na base vetorial."""
    matches = await vector_store.get_by_metadata(where={"paper_id": paper_id})
    return join_ordered_chunks(matches)[:max_chars]
