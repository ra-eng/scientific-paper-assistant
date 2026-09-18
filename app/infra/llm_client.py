from __future__ import annotations

from google import genai
from google.genai import errors, types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from app.core.settings import settings

_RETRYABLE_STATUS_CODES = {429, 503}


def _is_retryable(error: BaseException) -> bool:
    """429 (cota do free tier) e 503 (sobrecarga temporária do modelo) valem
    retry com backoff; qualquer outro erro da API (ex: 400 argumento inválido)
    é um bug e deve propagar imediatamente, não ser mascarado por retries."""
    return isinstance(error, errors.APIError) and error.code in _RETRYABLE_STATUS_CODES


_retry_on_rate_limit = retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)


class GeminiClient:
    """Wrapper assíncrono fino sobre o SDK google-genai (Gemini 2.0 Flash)."""

    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)

    @_retry_on_rate_limit
    async def generate(self, *, prompt: str, system_instruction: str | None = None) -> str:
        """Chamada single-shot, sem function calling e sem schema de saída."""
        response = await self._client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system_instruction),
        )
        return response.text or ""

    @_retry_on_rate_limit
    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: dict[str, object],
        system_instruction: str | None = None,
    ) -> str:
        """Chamada single-shot forçando saída em JSON validado contra
        `response_schema` (tipicamente `SomeModel.model_json_schema()`).
        Usada pelas tools do AnalystAgent, que precisam de saída estruturada
        (comparação, resumo, ranking) em vez de texto livre. Retorna o JSON
        cru como string; quem chama valida com `SomeModel.model_validate_json`."""
        response = await self._client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_json_schema=response_schema,
            ),
        )
        return response.text or ""

    @_retry_on_rate_limit
    async def generate_with_tools(
        self,
        *,
        contents: list[types.Content],
        tools: list[types.FunctionDeclaration],
        system_instruction: str | None = None,
    ) -> types.GenerateContentResponse:
        """Chamada com function calling manual: retorna a resposta crua do SDK
        para o agente decidir se há function_call parts a executar. O
        encadeamento de rodadas (reenviar o resultado da tool, repetir) é
        responsabilidade de quem chama , ver app/agents/rag_agent.py."""
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[types.Tool(function_declarations=tools)],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        return await self._client.aio.models.generate_content(
            model=settings.gemini_model,
            # list[Content] é um `contents` válido em runtime; o stub do SDK
            # declara o parâmetro como union de list invariante e não infere
            # list[Content] como subtipo compatível.
            contents=contents,  # type: ignore[arg-type]
            config=config,
        )
