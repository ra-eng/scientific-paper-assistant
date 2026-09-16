from __future__ import annotations

from pydantic import BaseModel

from app.ingest.parse import PaperSection


class Chunk(BaseModel):
    id: str
    paper_id: str
    section: str
    text: str


def _split_words(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    windows: list[str] = []
    start = 0
    while start < len(words):
        windows.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
        start += step
    return windows


def chunk_sections(sections: list[PaperSection], *, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Chunking section-aware: cada chunk pertence a exatamente uma seção,
    nunca cruza a fronteira entre seções (necessário para extract_section)."""
    chunks: list[Chunk] = []
    for section in sections:
        windows = _split_words(section.text, chunk_size=chunk_size, overlap=overlap)
        for index, window_text in enumerate(windows):
            chunk_id = f"{section.paper_id}:{section.section}:{index}"
            chunks.append(Chunk(id=chunk_id, paper_id=section.paper_id, section=section.section, text=window_text))
    return chunks
