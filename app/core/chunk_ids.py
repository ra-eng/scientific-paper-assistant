from __future__ import annotations


def chunk_index(chunk_id: str) -> int:
    """Extrai o índice de um chunk id no formato '{paper_id}:{section}:{index}'
    (ver app/ingest/chunk.py), usado para reconstruir a ordem original do texto."""
    return int(chunk_id.rsplit(":", 1)[-1])


def join_ordered_chunks(chunks: list[dict[str, str]]) -> str:
    """Reordena chunks (dicts com pelo menos 'id' e 'text') pela ordem
    original do texto e junta em uma única string."""
    ordered = sorted(chunks, key=lambda chunk: chunk_index(chunk["id"]))
    return "\n\n".join(chunk["text"] for chunk in ordered)
