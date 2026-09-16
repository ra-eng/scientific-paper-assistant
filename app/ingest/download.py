from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from app.core.papers import KNOWN_PAPERS as ARXIV_IDS

_ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"


async def download_paper(arxiv_id: str, *, destination_dir: str) -> str:
    """Baixa o PDF do arXiv e retorna o caminho local. Idempotente: não
    baixa de novo se o arquivo já existe (permite reexecutar `make setup`)."""
    destination = Path(destination_dir) / f"{arxiv_id}.pdf"
    if destination.exists():
        return str(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
        response = await client.get(_ARXIV_PDF_URL.format(arxiv_id=arxiv_id))
        response.raise_for_status()
        destination.write_bytes(response.content)

    return str(destination)


async def download_all(*, destination_dir: str) -> list[str]:
    """Baixa os 5 papers fixos do desafio em paralelo, na mesma ordem de ARXIV_IDS."""
    return await asyncio.gather(*(download_paper(arxiv_id, destination_dir=destination_dir) for arxiv_id in ARXIV_IDS))
