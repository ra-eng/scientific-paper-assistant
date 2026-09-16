from __future__ import annotations

import logging
from typing import TypedDict

from google.genai import types
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from app.agents._function_calling import extract_function_calls, extract_model_content, history_to_contents
from app.agents.analyst_agent import AnalystAgent
from app.agents.rag_agent import RAGAgent
from app.infra.llm_client import GeminiClient

logger = logging.getLogger(__name__)

_SYSTEM_INSTRUCTION = (
    "Você é o orquestrador de um sistema de análise de papers científicos de Machine "
    "Learning. Tem dois agentes disponíveis: consult_rag_agent, para perguntas sobre o "
    "conteúdo ou mecanismo de um paper (busca semântica, seções específicas); e "
    "consult_analyst_agent, para comparações, resumos executivos e rankings entre "
    "papers. Chame um ou os dois agentes conforme a pergunta exigir, com uma instrução "
    "clara e autocontida para cada um."
)


class _AgentInstructionParams(BaseModel):
    instruction: str = Field(..., description="Instrução autocontida para o agente executar")


_AGENT_PARAMS_SCHEMA = _AgentInstructionParams.model_json_schema()
_ROUTING_DECLARATIONS = [
    types.FunctionDeclaration(
        name="consult_rag_agent",
        description=(
            "Consulta o RAGAgent para busca semântica ou extração de seção específica de um "
            "paper. Use para perguntas sobre o conteúdo/mecanismo de um único paper."
        ),
        parameters_json_schema=_AGENT_PARAMS_SCHEMA,
    ),
    types.FunctionDeclaration(
        name="consult_analyst_agent",
        description="Consulta o AnalystAgent para comparar papers, resumir um paper ou ranqueá-los.",
        parameters_json_schema=_AGENT_PARAMS_SCHEMA,
    ),
]


class OrchestratorState(TypedDict):
    """Estado do grafo para uma única invocação; o histórico entre
    invocações vem do SQLite e é injetado em `history` a cada chamada."""

    question: str
    history: list[dict[str, str]]
    rag_instruction: str | None
    analyst_instruction: str | None
    rag_output: str | None
    analyst_output: str | None
    final_answer: str | None


class OrchestratorAgent:
    """Decide quais agentes acionar (function calling) e consolida a resposta final."""

    def __init__(self, llm_client: GeminiClient, rag_agent: RAGAgent, analyst_agent: AnalystAgent) -> None:
        self._llm_client = llm_client
        self._rag_agent = rag_agent
        self._analyst_agent = analyst_agent
        self._graph: CompiledStateGraph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(OrchestratorState)
        graph.add_node("decide", self._decide)
        graph.add_node("call_rag_agent", self._call_rag_agent)
        graph.add_node("call_analyst_agent", self._call_analyst_agent)
        graph.add_node("consolidate", self._consolidate)

        graph.add_edge(START, "decide")
        graph.add_conditional_edges("decide", self._route_after_decide)
        graph.add_edge("call_rag_agent", "consolidate")
        graph.add_edge("call_analyst_agent", "consolidate")
        graph.add_edge("consolidate", END)

        return graph.compile()

    async def ask(self, *, thread_id: str, question: str, history: list[dict[str, str]]) -> str:
        """Ponto de entrada usado pela camada de API. `thread_id` identifica
        a conversa; `history` já vem carregado do SQLite pela rota."""
        initial_state: OrchestratorState = {
            "question": question,
            "history": history,
            "rag_instruction": None,
            "analyst_instruction": None,
            "rag_output": None,
            "analyst_output": None,
            "final_answer": None,
        }
        final_state = await self._graph.ainvoke(initial_state)
        logger.info("thread=%s orquestrador concluiu a resposta", thread_id)
        return final_state["final_answer"] or ""

    async def _decide(self, state: OrchestratorState) -> dict[str, str | None]:
        contents = history_to_contents(state["history"])
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=state["question"])]))

        response = await self._llm_client.generate_with_tools(
            contents=contents,
            tools=_ROUTING_DECLARATIONS,
            system_instruction=_SYSTEM_INSTRUCTION,
        )
        model_content = extract_model_content(response)
        function_calls = extract_function_calls(model_content) if model_content is not None else []

        rag_instruction: str | None = None
        analyst_instruction: str | None = None
        for call in function_calls:
            args = call.args or {}
            if call.name == "consult_rag_agent":
                rag_instruction = args.get("instruction") or state["question"]
            elif call.name == "consult_analyst_agent":
                analyst_instruction = args.get("instruction") or state["question"]

        if rag_instruction is None and analyst_instruction is None:
            logger.warning("orquestrador não escolheu nenhum agente; usando RAGAgent como fallback")
            rag_instruction = state["question"]

        return {"rag_instruction": rag_instruction, "analyst_instruction": analyst_instruction}

    def _route_after_decide(self, state: OrchestratorState) -> list[str]:
        next_nodes = []
        if state["rag_instruction"] is not None:
            next_nodes.append("call_rag_agent")
        if state["analyst_instruction"] is not None:
            next_nodes.append("call_analyst_agent")
        return next_nodes

    async def _call_rag_agent(self, state: OrchestratorState) -> dict[str, str]:
        instruction = state["rag_instruction"] or state["question"]
        answer = await self._rag_agent.handle(instruction, context={"history": state["history"]})
        return {"rag_output": answer}

    async def _call_analyst_agent(self, state: OrchestratorState) -> dict[str, str]:
        instruction = state["analyst_instruction"] or state["question"]
        answer = await self._analyst_agent.handle(instruction, context={"history": state["history"]})
        return {"analyst_output": answer}

    async def _consolidate(self, state: OrchestratorState) -> dict[str, str]:
        if state["rag_output"] and state["analyst_output"]:
            prompt = (
                f"Pergunta original do usuário: {state['question']}\n\n"
                f"Resposta do agente de recuperação (RAG):\n{state['rag_output']}\n\n"
                f"Resposta do agente analista:\n{state['analyst_output']}\n\n"
                "Combine as duas em uma única resposta final, coerente e sem repetições."
            )
            final_answer = await self._llm_client.generate(prompt=prompt, system_instruction=_SYSTEM_INSTRUCTION)
        else:
            final_answer = state["rag_output"] or state["analyst_output"] or ""

        return {"final_answer": final_answer}
