"""Chama a API (já no ar) com as 5 perguntas de avaliação do enunciado
e imprime as respostas. Usado por `make run`."""

from __future__ import annotations

import asyncio

import httpx

_BASE_URL = "http://app:8000"
_MAX_STARTUP_ATTEMPTS = 10
_STARTUP_RETRY_DELAY_SECONDS = 2.0

QUESTIONS: list[str] = [
    "Qual é o mecanismo central proposto no paper Attention Is All You Need e como ele se diferencia de RNNs?",
    "Como o RAG combina recuperação e geração? Quais são suas limitações apontadas pelos autores?",
    "Compare a abordagem do ReAct com a do Toolformer para uso de ferramentas em LLMs.",
    (
        "Qual paper você considera mais relevante para construir um agente com uso de ferramentas "
        "externas? Justifique com base nos textos."
    ),
    "Faça um resumo executivo dos 5 papers em no máximo 5 bullet points cada.",
]


async def _wait_for_api(client: httpx.AsyncClient) -> None:
    """`docker compose up -d` retorna assim que o container sobe, não
    quando a API termina o startup (criação de tabelas etc). Tenta algumas
    vezes antes de desistir."""
    for attempt in range(1, _MAX_STARTUP_ATTEMPTS + 1):
        try:
            response = await client.get("/threads")
            response.raise_for_status()
            return
        except httpx.HTTPError:
            if attempt == _MAX_STARTUP_ATTEMPTS:
                raise
            await asyncio.sleep(_STARTUP_RETRY_DELAY_SECONDS)


async def main() -> None:
    async with httpx.AsyncClient(base_url=_BASE_URL, timeout=120.0) as client:
        await _wait_for_api(client)

        thread_response = await client.post("/threads")
        thread_response.raise_for_status()
        thread_id = thread_response.json()["thread_id"]

        for question in QUESTIONS:
            response = await client.post(f"/threads/{thread_id}/messages", json={"content": question})
            response.raise_for_status()
            answer = response.json()["response"]
            print(f"\n=== {question} ===\n{answer}\n")


if __name__ == "__main__":
    asyncio.run(main())
