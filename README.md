# Assistente de Análise de Artigos Científicos

Sistema multi-agente que responde perguntas analíticas sobre 5 papers de Machine Learning fixos, exposto como API REST em FastAPI, 100% dockerizado. Solução do desafio técnico de Data Science da Winnin (ver enunciado em `CHALLANGE/desafio/DATASCI.md`).

Documentos de apoio: [`PLAN.md`](./PLAN.md) (roadmap completo) e [`FRAMEWORKS.md`](./FRAMEWORKS.md) (comparação LangChain/LangGraph/LlamaIndex/CrewAI/AutoGen e justificativa da escolha).

---

## 1. Visão geral da arquitetura

```
Usuário
   │  POST /threads/{id}/messages
   ▼
┌──────────────────────────────┐
│      API (FastAPI)           │  valida input, carrega histórico da thread (SQLite), chama o orquestrador
└───────────────┬───────────────┘
                ▼
┌────────────────────────────────────────────────────┐
│                 OrchestratorAgent                    │   grafo LangGraph
│                                                        │
│   decide ──▶ [call_rag_agent? / call_analyst_agent?] ──▶ consolidate
└──────┬────────────────────────┬──────────────────────┘
       ▼                        ▼
┌───────────────┐      ┌──────────────────────┐
│    RAGAgent     │      │     AnalystAgent      │
│ • search_documents │  │ • compare_papers       │
│ • extract_section  │  │ • summarize             │
│                     │  │ • rank_papers            │
└──────┬──────────┘      └──────────┬────────────┘
       ▼                            ▼
┌────────────────────────────────────────────────┐
│  ChromaDB (vector store)  +  Gemini 2.0 Flash    │
└──────────────────────────────────────────────────┘

Camada transversal: SQLite (threads + mensagens), via SQLAlchemy async
```

**Fluxo de uma pergunta:**
1. `POST /threads/{id}/messages` carrega o histórico da thread do SQLite e chama `OrchestratorAgent.ask(...)`.
2. O grafo LangGraph roda o nó `decide`: manda a pergunta (+ histórico) pro Gemini com duas *function declarations* sintéticas — `consult_rag_agent` e `consult_analyst_agent`. O modelo decide qual(is) acionar.
3. Os nós `call_rag_agent`/`call_analyst_agent` rodam (em paralelo quando os dois são escolhidos) delegando para `RAGAgent.handle(...)`/`AnalystAgent.handle(...)`.
4. **Cada agente roda seu próprio loop de function calling** contra o Gemini, decidindo quais das suas tools exclusivas encadear (ex: `RAGAgent` chama `search_documents`, olha o resultado, decide se chama `extract_section` também, etc.) — é aqui que os **dois níveis de function calling** do desafio se encontram: o orquestrador decide o agente, o agente decide a tool.
5. O nó `consolidate` retorna a resposta direto se só um agente respondeu, ou pede ao Gemini uma síntese quando os dois responderam.
6. A resposta final é persistida no SQLite e devolvida ao usuário.

Ingestão (`make setup`, roda antes de tudo): baixa os 5 PDFs do arXiv → extrai texto por seção (`pdfplumber`) → chunking section-aware → embeddings locais (`sentence-transformers`) → upsert no ChromaDB.

---

## 2. Distinção entre tools e agentes

Reflete a arquitetura pedida no enunciado, implementada em código (não só em nomenclatura):

- **`Tool[ParamsT]`** (`app/tools/base.py`) — classe abstrata genérica. Toda tool tem `name`, `description`, um `params_model: type[BaseModel]` (schema tipado, vira `FunctionDeclaration` do Gemini automaticamente) e um único método `async def run(params) -> ToolResult`. **Sem estado, sem decisão**: uma tool nunca decide se deve rodar — quem decide é o agente dono dela. `ToolResult` é o envelope padronizado (`success`, `data`, `error`) que toda tool devolve, nunca uma exceção crua — os limites de tool sempre capturam erros esperados (falha de rede no vector store, JSON inválido do LLM, argumentos inválidos) e os traduzem em `ToolResult(success=False, ...)`.

  5 tools, cada uma dona de exatamente um agente: `search_documents`/`extract_section` (`RAGAgent`) e `compare_papers`/`summarize`/`rank_papers` (`AnalystAgent`). Nenhuma tool é compartilhada entre agentes.

- **`Agent`** (`app/agents/base.py`) — classe abstrata com um único método `async def handle(instruction, *, context) -> str`. Um agente **tem contexto** (recebe histórico da thread), **decide** quais das suas próprias tools encadear via function calling do Gemini, e **relata** ao orquestrador com um texto consolidado — nunca um `ToolResult` bruto. `RAGAgent` e `AnalystAgent` compartilham o loop de decisão (`app/agents/_function_calling.py`) porque a mecânica é idêntica; o que muda entre eles é só o conjunto de tools e o `system_instruction`.

- **`OrchestratorAgent`** (`app/agents/orchestrator.py`) — não tem tools próprias. Sua única responsabilidade é decidir *quais agentes* acionar (via um grafo LangGraph, não um loop manual) e consolidar as respostas. Ele nunca chama uma tool diretamente — só conversa com `RAGAgent`/`AnalystAgent` através do método `handle`.

Em suma: **tool = capacidade atômica sem julgamento; agente = julgamento + memória de contexto + tools próprias; orquestrador = julgamento sobre agentes, sem tools.**

---

## 3. Instruções de setup

Pré-requisito: Docker + Docker Compose. Nenhuma outra dependência local é necessária (tudo roda em container).

```bash
git clone <repo> && cd datasci-agent
cp .env.example .env      # preencher GEMINI_API_KEY (grátis em aistudio.google.com)

make setup   # baixa os 5 PDFs, chunka, embeda localmente e popula o ChromaDB
make run     # sobe os containers e executa as 5 perguntas de avaliação via API
make test    # roda a suíte de testes (unitários + integração)
make down    # derruba os containers
```

Documentação interativa (Swagger) em `http://localhost:8000/docs` depois de `make run`.

### Rodando fora do Docker (desenvolvimento local)

```bash
pip install -e ".[dev]"
cp .env.example .env   # preencher GEMINI_API_KEY; ajustar CHROMA_HOST=localhost se o Chroma rodar via docker run avulso
python -m scripts.setup
uvicorn app.main:app --reload
```

---

## 4. Decisões técnicas

| Decisão | Escolha | Por quê |
|---|---|---|
| **Framework de orquestração** | **LangGraph** para o roteamento orquestrador→agentes; SDK `google-genai` cru para as chamadas de LLM dentro de tools/agentes | LangGraph é uma lib de baixo nível (grafo de estados explícito), não uma camada de "agente mágico" — dá controle fino sobre o roteamento (`add_conditional_edges` com fan-out pra rodar RAG e Analyst em paralelo quando os dois são necessários) sem competir com as classes `Tool`/`Agent`/`ToolResult` próprias exigidas pelo desafio. Comparação completa em [`FRAMEWORKS.md`](./FRAMEWORKS.md). |
| **Vector store** | **ChromaDB** (container HTTP próprio) | Filtro por metadata (`paper_id`, `section`) nativo — essencial pro `extract_section` e pro grounding do `AnalystAgent` — sem precisar reimplementar indexação de metadata como seria necessário com FAISS. |
| **Chunking** | **Section-aware** (parseia a estrutura do paper, chunka dentro de cada seção com janela de palavras + overlap) | `extract_section` depende de saber onde cada seção começa/termina; um splitter genérico por tamanho fixo cortaria seções ao meio e perderia essa informação. |
| **Modelo de embedding** | **`sentence-transformers` local** (`all-MiniLM-L6-v2`), não a API de embedding do Gemini | Mantém a ingestão (`make setup`) reproduzível offline, sem depender de quota de API só pra popular o vector store — reserva as chamadas ao Gemini para geração e function calling, que é o que está sendo avaliado. |
| **Parsing de PDF** | **`pdfplumber`**, com detecção de colunas por análise de "vão" horizontal (não corte fixo na metade da página) e filtro de caracteres não-verticais | Os 5 papers têm layout de duas colunas; um corte ingênuo no meio da página fatia palavras ao meio quando uma delas cai sobre a linha de corte (bug real encontrado durante o desenvolvimento — documentado no histórico do projeto). A abordagem final agrupa palavras por posição e só separa colunas onde há um vão real de texto. Também filtra o carimbo vertical do arXiv (`arXiv:...v7 [cs.CL]...`), que embaralhava o bloco de autores. |
| **Saída estruturada do AnalystAgent** | `response_mime_type="application/json"` + `response_json_schema` (gerado direto de `model_json_schema()` dos modelos Pydantic de saída) | Em vez de pedir texto livre e fazer parsing manual (regex/heurística) pra extrair bullets/ranking, o Gemini já devolve JSON validável contra o schema — elimina uma classe inteira de bugs de parsing frágil. |
| **Function calling em dois níveis** | Orquestrador decide *agentes* via function calling; cada agente decide *tools* via seu próprio loop de function calling | Loop genérico compartilhado (`app/agents/_function_calling.py`) entre `RAGAgent`/`AnalystAgent`; o orquestrador usa a mesma mecânica de baixo nível (`GeminiClient.generate_with_tools`) mas dentro do grafo LangGraph, já que suas "tools" são agentes, não `Tool` instances. |
| **Memória por thread** | **SQLite via SQLAlchemy async** (`aiosqlite`), não o checkpointer nativo do LangGraph | Requisito explícito do enunciado. O grafo do orquestrador é stateless por invocação — o histórico entra como parâmetro (`history`) carregado do SQLite pela rota, evitando duas fontes de verdade para a mesma informação. |
| **Injeção de dependências** | `GeminiClient`/`VectorStoreClient`/`EmbeddingClient` e os agentes como singletons via `@lru_cache` (`app/api/dependencies.py`); `ThreadRepository` por request | Os clients por baixo mantêm conexão HTTP (Chroma) ou modelo carregado em memória (sentence-transformers) — recriar isso a cada request seria caro e desnecessário. A sessão de banco, por outro lado, precisa ser por request (escopo transacional). |

---

## 5. Limitações conhecidas

- **`extract_section` é impreciso em papers com títulos de seção não-padrão.** A heurística de detecção de seção usa uma lista fixa de nomes conhecidos (abstract, introduction, method, results, conclusion...). Papers como ReAct/Toolformer têm seções com títulos customizados (ex: "2 ReAct: Synergizing Reasoning and Acting") que acabam absorvidos pela seção anterior conhecida. Não afeta `search_documents` (busca semântica não depende do rótulo de seção), só reduz a precisão de extrações pontuais por seção nesses papers específicos.
- **Bloco de autores/título da primeira página pode sair com ordem levemente incorreta** em papers com layout complexo na capa (múltiplas colunas de afiliação, notas de rodapé). O corpo do texto (o que importa para responder as perguntas do desafio) não é afetado.
- **"100% local" é parcial por exigência do próprio enunciado**: a ingestão e o vector store rodam inteiramente offline, mas as chamadas ao Gemini (geração e function calling) são, por definição, uma API externa.
- **Sem retry/backoff para rate limit do Gemini free tier.** Chamadas concorrentes (`asyncio.gather` no `AnalystAgent` ao buscar contexto de múltiplos papers) podem esbarrar em cota; não há lógica de retry automático hoje.
- **`GET /threads` sem paginação.** Aceitável para o escopo do desafio (poucas threads de teste); não escalaria para um volume grande de conversas.
- **Sem autenticação na API.** Fora do escopo do desafio, mas seria necessário antes de qualquer uso além de avaliação local.
- **Sem CI configurado.** `ruff`, `mypy` e `pytest` rodam localmente (`make test` cobre os testes; lint/type-check são manuais) — não há workflow de GitHub Actions automatizando isso a cada push.
- **FAISS não foi implementado como alternativa ao ChromaDB** — o enunciado permite qualquer um dos dois; só o Chroma foi construído, pela razão explicada na seção 4.
