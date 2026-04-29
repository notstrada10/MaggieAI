"""Abstract `InferenceClient` interface.

Each implementation (Claude, local MLX, remote vLLM, ...) exposes the
same `generate` method. The `Router` chooses which client to invoke
based on the task kind (see `router.py`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Literal


class TaskKind(str, Enum):
    """Task kind — used by the router to pick the backend.

    - ROUTING: fast classification, short output. Local.
    - LIGHTWEIGHT: helper / disambiguation. Local.
    - TRANSLATION: critical translation with rationale. Claude (cloud).
    - CRITIQUE: self-critique on the draft. Claude (cloud).
    """

    ROUTING = "routing"
    LIGHTWEIGHT = "lightweight"
    TRANSLATION = "translation"
    CRITIQUE = "critique"


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class GenerationRequest:
    messages: list[Message]
    max_tokens: int = 1024
    temperature: float = 0.2
    json_mode: bool = False
    """If True the client must return valid JSON (best-effort)."""


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class InferenceClient(ABC):
    """Contract every backend must satisfy."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse: ...

    async def aclose(self) -> None:
        """Resource cleanup — override if the client holds HTTP connections."""
        return None
