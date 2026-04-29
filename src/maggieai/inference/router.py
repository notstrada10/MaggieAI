"""Router that picks the right `InferenceClient` for each task.

Default strategy — HYBRID (local + Claude):
- ROUTING / LIGHTWEIGHT  → MlxLocalClient (Qwen-2.5-14B on MLX)
- TRANSLATION / CRITIQUE → ClaudeApiClient (claude-sonnet-4-6)

Can be overridden via env / constructor to push everything to local
(mode='local-only') or to Claude (mode='claude-only').
"""

from __future__ import annotations

import logging
from typing import Literal

from maggieai.inference.claude_api import ClaudeApiClient
from maggieai.inference.client import (
    GenerationRequest,
    GenerationResponse,
    InferenceClient,
    TaskKind,
)
from maggieai.inference.mlx_local import MlxLocalClient

logger = logging.getLogger(__name__)

RoutingMode = Literal["hybrid", "local-only", "claude-only"]


class InferenceRouter:
    def __init__(
        self,
        mode: RoutingMode = "hybrid",
        local: InferenceClient | None = None,
        claude: InferenceClient | None = None,
    ) -> None:
        self.mode: RoutingMode = mode
        self._local = local
        self._claude = claude

    def _local_client(self) -> InferenceClient:
        if self._local is None:
            self._local = MlxLocalClient()
        return self._local

    def _claude_client(self) -> InferenceClient:
        if self._claude is None:
            self._claude = ClaudeApiClient()
        return self._claude

    def for_task(self, task: TaskKind) -> InferenceClient:
        if self.mode == "local-only":
            return self._local_client()
        if self.mode == "claude-only":
            return self._claude_client()
        # hybrid
        if task in (TaskKind.TRANSLATION, TaskKind.CRITIQUE):
            return self._claude_client()
        return self._local_client()

    async def generate(self, task: TaskKind, request: GenerationRequest) -> GenerationResponse:
        client = self.for_task(task)
        logger.debug("Inference: task=%s client=%s", task.value, client.name)
        return await client.generate(request)

    async def aclose(self) -> None:
        for c in (self._local, self._claude):
            if c is not None:
                await c.aclose()
