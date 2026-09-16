"""Pipeline de ingestão completo — download, parse, chunk, embed, upsert.
Chamado por `make setup`, roda dentro do container `app`."""

from __future__ import annotations

import asyncio
import logging

from app.core.logging import configure_logging
from app.infra.embeddings import EmbeddingClient
from app.infra.vector_store import VectorStoreClient
from app.ingest.chunk import Chunk, chunk_sections
from app.ingest.download import ARXIV_IDS, download_all
from app.ingest.embed import embed_and_upsert
from app.ingest.parse import extract_sections

_PAPERS_DIR = "data/papers"

logger = logging.getLogger(__name__)


async def main() -> None:
    configure_logging()

    logger.info("Baixando %d papers do arXiv...", len(ARXIV_IDS))
    pdf_paths = await download_all(destination_dir=_PAPERS_DIR)

    logger.info("Extraindo seções e chunkando...")
    all_chunks: list[Chunk] = []
    for arxiv_id, pdf_path in zip(ARXIV_IDS, pdf_paths, strict=True):
        sections = extract_sections(pdf_path, paper_id=arxiv_id)
        all_chunks.extend(chunk_sections(sections))

    logger.info("Gerando embeddings e populando o ChromaDB (%d chunks)...", len(all_chunks))
    vector_store = VectorStoreClient()
    embedding_client = EmbeddingClient()
    await embed_and_upsert(all_chunks, vector_store=vector_store, embedding_client=embedding_client)

    logger.info("Ingestão concluída.")


if __name__ == "__main__":
    asyncio.run(main())
