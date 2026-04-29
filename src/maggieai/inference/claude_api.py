"""Client Anthropic Claude per i nodi critici del reasoning loop.

Usato per `draft_translation` e `self_critique` — task dove la qualità
del ragionamento (e dell'italiano) supera nettamente quella di un
modello locale 14B.
"""

from __future__ import annotations

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam

from maggieai.config import get_settings
from maggieai.inference.client import (
    GenerationRequest,
    GenerationResponse,
    InferenceClient,
    Message,
)


class ClaudeApiClient(InferenceClient):
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        key = api_key or settings.anthropic_api_key
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY non configurato — impostalo in .env "
                "oppure passa api_key esplicito al costruttore."
            )
        self._model = model or settings.claude_model
        self._client = AsyncAnthropic(api_key=key)

    @property
    def name(self) -> str:
        return f"claude:{self._model}"

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        system_prompt, conv = _split_system(request.messages)
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system=system_prompt or "",
            messages=conv,
        )
        # Concatena tutti i blocchi text del response
        text = "".join(block.text for block in response.content if block.type == "text")
        return GenerationResponse(
            text=text,
            model=self._model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def aclose(self) -> None:
        await self._client.close()


def _split_system(messages: list[Message]) -> tuple[str | None, list[MessageParam]]:
    """Anthropic vuole il system prompt come parametro top-level, non in `messages`."""
    system: str | None = None
    conv: list[MessageParam] = []
    for m in messages:
        if m.role == "system":
            system = m.content if system is None else f"{system}\n\n{m.content}"
        else:
            conv.append({"role": m.role, "content": m.content})
    return system, conv
