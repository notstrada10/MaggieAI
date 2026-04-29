"""Interfaccia astratta `InferenceClient`.

Ogni implementazione (Claude, MLX locale, vLLM remoto, ...) espone lo
stesso metodo `generate`. Il `Router` sceglie quale client invocare
in base al tipo di task (vedi `router.py`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Literal


class TaskKind(str, Enum):
    """Tipologia del task — usata dal router per scegliere il backend.

    - ROUTING: classificazione veloce, output corto. Locale.
    - LIGHTWEIGHT: helper / disambiguazione. Locale.
    - TRANSLATION: traduzione critica con rationale. Claude (cloud).
    - CRITIQUE: self-critique sul draft. Claude (cloud).
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
    """Se True il client deve restituire un JSON valido (best-effort)."""


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class InferenceClient(ABC):
    """Contratto che ogni backend deve rispettare."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse: ...

    async def aclose(self) -> None:
        """Chiusura risorse — override se il client tiene connessioni HTTP."""
        return None
