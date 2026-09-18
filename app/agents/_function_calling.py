from __future__ import annotations

import logging
from typing import Any

from google.genai import types
from pydantic import ValidationError

from app.infra.llm_client import GeminiClient
from app.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)

_MAX_TOOL_CALL_ROUNDS = 4
_HISTORY_ROLE_MAP = {"user": "user", "assistant": "model"}


def tool_to_function_declaration(tool: Tool[Any]) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=tool.name,
        description=tool.description,
        parameters_json_schema=tool.params_model.model_json_schema(),
    )


def history_to_contents(history: list[dict[str, str]]) -> list[types.Content]:
    return [
        types.Content(
            role=_HISTORY_ROLE_MAP.get(turn["role"], "user"),
            parts=[types.Part.from_text(text=turn["content"])],
        )
        for turn in history
    ]


def extract_model_content(response: types.GenerateContentResponse) -> types.Content | None:
    """`candidates`/`content` são Optional na API , o Gemini pode devolver
    uma resposta sem candidatos quando bloqueia por segurança. `None` aqui
    sinaliza esse caso para quem chamou, em vez de estourar um IndexError."""
    candidates = response.candidates or []
    if not candidates or candidates[0].content is None:
        return None
    return candidates[0].content


def extract_function_calls(content: types.Content) -> list[types.FunctionCall]:
    return [part.function_call for part in (content.parts or []) if part.function_call is not None]


async def run_tool_calling_loop(
    *,
    llm_client: GeminiClient,
    tools: tuple[Tool[Any], ...],
    system_instruction: str,
    instruction: str,
    history: list[dict[str, str]],
    agent_name: str,
) -> str:
    """Loop genérico de function calling: manda a instrução (+ histórico da
    thread), executa as tool calls que vierem, reenvia o resultado, repete
    até a resposta ser só texto ou até o limite de rodadas. Compartilhado
    entre RAGAgent e AnalystAgent , a diferença entre eles é só o conjunto
    de tools e o system_instruction, não a lógica de orquestração em si."""
    tool_by_name = {tool.name: tool for tool in tools}
    function_declarations = [tool_to_function_declaration(tool) for tool in tools]

    contents = history_to_contents(history)
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=instruction)]))

    response: types.GenerateContentResponse | None = None
    for _ in range(_MAX_TOOL_CALL_ROUNDS):
        response = await llm_client.generate_with_tools(
            contents=contents,
            tools=function_declarations,
            system_instruction=system_instruction,
        )
        model_content = extract_model_content(response)
        if model_content is None:
            logger.warning(
                "%s: resposta do Gemini sem conteúdo utilizável (possível bloqueio de segurança)", agent_name
            )
            return ""

        function_calls = extract_function_calls(model_content)
        if not function_calls:
            return response.text or ""

        contents.append(model_content)
        response_parts = [await _execute_tool_call(call, tool_by_name) for call in function_calls]
        contents.append(types.Content(role="user", parts=response_parts))

    logger.warning(
        "%s atingiu o limite de %d rodadas de tool calling sem resposta final", agent_name, _MAX_TOOL_CALL_ROUNDS
    )
    return response.text or "" if response is not None else ""


async def _execute_tool_call(function_call: types.FunctionCall, tool_by_name: dict[str, Tool[Any]]) -> types.Part:
    function_name = function_call.name or "unknown"
    tool = tool_by_name.get(function_name)
    if tool is None:
        result = ToolResult(success=False, error=f"Tool desconhecida: {function_name}")
    else:
        try:
            params = tool.params_model.model_validate(function_call.args or {})
        except ValidationError as error:
            result = ToolResult(success=False, error=f"Argumentos inválidos para {function_name}: {error}")
        else:
            result = await tool.run(params)

    return types.Part.from_function_response(name=function_name, response=result.model_dump())
