"""State of the LangGraph reasoning loop.

A `TypedDict` because LangGraph needs it for the automatic merge between
nodes. Each node receives a State and returns a *delta* (Pydantic
dataclass or partial dict): LangGraph performs the merge.
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
    """State shared across all nodes of the reasoning loop."""

    # Input ------------------------------------------------------------
    trace_id: UUID
    input_text: str

    # Output of each step ---------------------------------------------
    morpho_analysis: SentenceAnalysis
    phenomena: list[str]
    tm_hits: list[TMHit]
    grammar_hits: list[GrammarHit]

    # Generation + critique -------------------------------------------
    draft_translation: str
    draft_rationale: str
    critique: str
    issues_found: bool
    iterations: int

    # Final output ----------------------------------------------------
    output: dict[str, Any]
