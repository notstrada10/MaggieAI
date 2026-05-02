"""DeepSeek client (OpenAI-compatible chat-completions API).

DeepSeek's API is OpenAI-compatible, so this client is structurally
similar to `MlxLocalClient` — only the base URL changes, plus a
Bearer token for auth.

Available models as of May 2026:
- `deepseek-v4-pro`: flagship MoE (1.6T params, 49B activated), strongest
- `deepseek-v4-flash`: lighter MoE (284B params, 13B activated), cheaper
"""

from __future__ import annotations

import httpx

from maggieai.config import get_settings
from maggieai.inference.client import (
    GenerationRequest,
    GenerationResponse,
    InferenceClient,
)


class DeepSeekApiClient(InferenceClient):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        settings = get_settings()
        key = api_key or settings.deepseek_api_key
        if not key:
            raise ValueError(
                "DEEPSEEK_API_KEY not configured — set it in .env or pass an "
                "explicit api_key to the constructor."
            )
        self._model = model or settings.deepseek_model
        self._base_url = (base_url or settings.deepseek_base_url).rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {key}"},
        )

    @property
    def name(self) -> str:
        return f"deepseek:{self._model}"

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.json_mode:
            payload["response_format"] = {"type": "json_object"}

        resp = await self._client.post(f"{self._base_url}/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return GenerationResponse(
            text=choice["message"]["content"],
            model=self._model,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
