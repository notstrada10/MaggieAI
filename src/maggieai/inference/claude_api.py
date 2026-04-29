"""Anthropic Claude client for the critical reasoning-loop nodes.

Used for `draft_translation` and `self_critique` — tasks where the
quality of reasoning (and of the target language) clearly outranks a
local 14B model.
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
                "ANTHROPIC_API_KEY not configured — set it in .env or pass an "
                "explicit api_key to the constructor."
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
        # Concatenate all text blocks of the response
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
    """Anthropic expects the system prompt as a top-level parameter, not in `messages`."""
    system: str | None = None
    conv: list[MessageParam] = []
    for m in messages:
        if m.role == "system":
            system = m.content if system is None else f"{system}\n\n{m.content}"
        else:
            conv.append({"role": m.role, "content": m.content})
    return system, conv
