from __future__ import annotations

KNOWN_PAPERS: dict[str, str] = {
    "1706.03762": "Attention Is All You Need",
    "1810.04805": "BERT: Pre-training of Deep Bidirectional Transformers",
    "2005.11401": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "2210.03629": "ReAct: Synergizing Reasoning and Acting in Language Models",
    "2302.04761": "Toolformer: Language Models Can Teach Themselves to Use Tools",
}


def format_known_papers() -> str:
    """Catálogo fechado formatado para injeção em system_instruction de agentes —
    sem isso, um agente não tem como saber quais arXiv IDs existem na base vetorial
    quando o usuário não os menciona explicitamente na pergunta."""
    return "\n".join(f"- {arxiv_id}: {title}" for arxiv_id, title in KNOWN_PAPERS.items())
