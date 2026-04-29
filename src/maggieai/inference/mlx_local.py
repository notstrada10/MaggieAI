"""Client for the local MLX server (OpenAI-compatible format).

The actual server runs NATIVELY on macOS via `mlx_lm.server` (see
`inference/server.py` and README §"Native inference"). This client
reaches it over HTTP — so it can also point to vLLM or llama.cpp with
the same interface, no code changes required.
"""

from __future__ import annotations

import httpx

from maggieai.config import get_settings
from maggieai.inference.client import (
    GenerationRequest,
    GenerationResponse,
    InferenceClient,
)


class MlxLocalClient(InferenceClient):
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        settings = get_settings()
        self._base_url = (base_url or settings.inference_local_url).rstrip("/")
        self._model = model or settings.inference_local_model
        self._client = httpx.AsyncClient(timeout=timeout)

    @property
    def name(self) -> str:
        return f"mlx-local:{self._model}"

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.json_mode:
            # mlx-lm.server supports `response_format` like the OpenAI API
            payload["response_format"] = {"type": "json_object"}

        resp = await self._client.post(f"{self._base_url}/v1/chat/completions", json=payload)
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
