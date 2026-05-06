"""DeepSeek client (OpenAI-compatible chat-completions API).

DeepSeek's API is OpenAI-compatible, so this client is structurally
similar to `MlxLocalClient` — only the base URL changes, plus a
Bearer token for auth.

Available models as of May 2026:
- `deepseek-v4-pro`: flagship MoE (1.6T params, 49B activated), strongest
- `deepseek-v4-flash`: lighter MoE (284B params, 13B activated), cheaper
"""

from __future__ import annotations

import asyncio
import logging
import random

import httpx

from maggieai.config import get_settings
from maggieai.inference.client import (
    GenerationRequest,
    GenerationResponse,
    InferenceClient,
)

logger = logging.getLogger(__name__)

# Errors that warrant a retry. Pure transport-level failures (DNS, TCP,
# read timeouts, mid-request disconnects) — NOT 4xx auth/validation
# errors, which are deterministic and won't get better on retry.
_RETRYABLE_TRANSPORT_ERRORS: tuple[type[Exception], ...] = (
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
)


class DeepSeekApiClient(InferenceClient):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 300.0,
        max_retries: int = 3,
        retry_initial_delay: float = 1.0,
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
        self._max_retries = max_retries
        self._retry_initial_delay = retry_initial_delay
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

        resp = await self._post_with_retry("/chat/completions", payload)
        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return GenerationResponse(
            text=choice["message"]["content"],
            model=self._model,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )

    async def _post_with_retry(self, path: str, payload: dict[str, object]) -> httpx.Response:
        """POST with bounded exponential backoff on transient failures.

        Observed in production: Docker's embedded DNS resolver intermittently
        fails to resolve `api.deepseek.com` under sustained load, surfacing as
        `httpx.ConnectError`. A single retry usually clears it; we cap at
        three attempts so a genuinely-down API still fails fast.
        """
        url = f"{self._base_url}{path}"
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._client.post(url, json=payload)
                resp.raise_for_status()
                return resp
            except _RETRYABLE_TRANSPORT_ERRORS as exc:
                last_exc = exc
            except httpx.HTTPStatusError as exc:
                # 4xx is deterministic (auth, validation) — don't retry.
                if exc.response.status_code < 500 and exc.response.status_code != 429:
                    raise
                last_exc = exc
            if attempt == self._max_retries:
                break
            delay = self._retry_initial_delay * (2**attempt) + random.uniform(0, 0.5)
            logger.warning(
                "DeepSeek %s failed (attempt %d/%d): %s — retrying in %.2fs",
                path,
                attempt + 1,
                self._max_retries + 1,
                type(last_exc).__name__,
                delay,
            )
            await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

    async def aclose(self) -> None:
        await self._client.aclose()
