"""State del reasoning loop LangGraph.

Una `TypedDict` perché LangGraph ne ha bisogno per il merge automatico
fra nodi. Ogni nodo riceve uno State e ne restituisce un *delta*
(Pydantic dataclass o dict parziale): LangGraph fa il merge.
"""

from __future__ import annotations

from typing import Any, TypedDict
from uuid import UUID

from maggieai.morphology.pipeline import SentenceAnalysis


class TMHit(TypedDict):
    source_text: str
    target_text: str
    author: str | None
    work: str | None
    locator: str | None
    translator: str | None
    distance: float


class GrammarHit(TypedDict):
    phenomenon: str
    description: str
    examples: list[dict[str, Any]]
    source: str | None


class AgentState(TypedDict, total=False):
    """State condiviso fra tutti i nodi del reasoning loop."""

    # Input ------------------------------------------------------------
    trace_id: UUID
    input_text: str

    # Output dei vari step --------------------------------------------
    morpho_analysis: SentenceAnalysis
    phenomena: list[str]
    tm_hits: list[TMHit]
    grammar_hits: list[GrammarHit]

    # Generazione + critique ------------------------------------------
    draft_translation: str
    draft_rationale: str
    critique: str
    issues_found: bool
    iterations: int

    # Output finale ---------------------------------------------------
    output: dict[str, Any]
