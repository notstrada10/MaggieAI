"""Nodi del reasoning loop LangGraph.

Ogni funzione `(state) -> partial_state` è isolatamente testabile.
LangGraph farà il merge automatico nel `AgentState` complessivo.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from maggieai.agent import prompts
from maggieai.agent.state import AgentState
from maggieai.agent.tools import all_grammar_patterns, grammar_lookup, tm_lookup
from maggieai.config import get_settings
from maggieai.inference.client import GenerationRequest, Message, TaskKind
from maggieai.inference.router import InferenceRouter
from maggieai.morphology.phenomena import detect as detect_phenomena
from maggieai.morphology.pipeline import SentenceAnalysis

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 2


# -------------------------------------------------------------------
# 1. morpho_parse — chiamata HTTP al servizio morphology
# -------------------------------------------------------------------
async def morpho_parse(state: AgentState) -> dict[str, Any]:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{settings.morphology_url}/analyze",
            json={"text": state["input_text"]},
        )
        resp.raise_for_status()
        analysis = SentenceAnalysis.model_validate(resp.json())
    return {"morpho_analysis": analysis}


# -------------------------------------------------------------------
# 2. phenomena_detect — pure-Python su feature CLTK + pattern DB
# -------------------------------------------------------------------
def phenomena_detect(state: AgentState) -> dict[str, Any]:
    rules = all_grammar_patterns()
    detected = detect_phenomena(state["morpho_analysis"], rules)
    return {"phenomena": detected}


# -------------------------------------------------------------------
# 3. retrieve — Tool A (TM via pgvector) + Tool B (grammar lookup)
# -------------------------------------------------------------------
async def retrieve(state: AgentState) -> dict[str, Any]:
    # Embedding del testo originale per la ricerca semantica
    from maggieai.ingestion.embedder import embed
    [vec] = embed([state["input_text"]])

    tm_hits = tm_lookup(vec, k=5)
    g_hits = grammar_lookup(state.get("phenomena", []))
    return {"tm_hits": tm_hits, "grammar_hits": g_hits}


# -------------------------------------------------------------------
# 4. draft_translation — LLM (Claude) con tutte le evidence
# -------------------------------------------------------------------
async def draft_translation(state: AgentState, router: InferenceRouter) -> dict[str, Any]:
    user = prompts.render(
        "draft_translation.j2",
        input_text=state["input_text"],
        morpho=state["morpho_analysis"].model_dump(),
        phenomena=state.get("phenomena", []),
        tm_hits=state.get("tm_hits", []),
        grammar_hits=state.get("grammar_hits", []),
        previous_draft=state.get("draft_translation"),
        previous_critique=state.get("critique"),
    )
    system = prompts.render("system_translator.j2")
    response = await router.generate(
        TaskKind.TRANSLATION,
        GenerationRequest(
            messages=[Message("system", system), Message("user", user)],
            max_tokens=1500,
            temperature=0.2,
            json_mode=True,
        ),
    )
    parsed = _parse_json(response.text)
    return {
        "draft_translation": parsed.get("translation", ""),
        "draft_rationale": parsed.get("rationale", ""),
        "iterations": state.get("iterations", 0) + 1,
    }


# -------------------------------------------------------------------
# 5. self_critique — verifica del draft contro grammar evidence
# -------------------------------------------------------------------
async def self_critique(state: AgentState, router: InferenceRouter) -> dict[str, Any]:
    user = prompts.render(
        "self_critique.j2",
        input_text=state["input_text"],
        draft=state["draft_translation"],
        rationale=state["draft_rationale"],
        morpho=state["morpho_analysis"].model_dump(),
        grammar_hits=state.get("grammar_hits", []),
    )
    system = prompts.render("system_critic.j2")
    response = await router.generate(
        TaskKind.CRITIQUE,
        GenerationRequest(
            messages=[Message("system", system), Message("user", user)],
            max_tokens=800,
            temperature=0.0,
            json_mode=True,
        ),
    )
    parsed = _parse_json(response.text)
    return {
        "critique": parsed.get("critique", ""),
        "issues_found": bool(parsed.get("issues_found", False)),
    }


# -------------------------------------------------------------------
# 6. format_output — costruisce il JSON finale per il client
# -------------------------------------------------------------------
def format_output(state: AgentState) -> dict[str, Any]:
    citations: list[dict[str, Any]] = []
    for hit in state.get("tm_hits", [])[:3]:
        citations.append({
            "type": "translation_memory",
            "source": f"{hit['author'] or '?'} {hit['work'] or ''} {hit['locator'] or ''}".strip(),
            "translator": hit["translator"],
            "distance": hit["distance"],
        })
    for hit in state.get("grammar_hits", []):
        citations.append({
            "type": "grammar_rule",
            "rule": hit["phenomenon"],
            "source": hit["source"],
        })

    output = {
        "translation": state.get("draft_translation", ""),
        "rationale": state.get("draft_rationale", ""),
        "morpho_analysis": state["morpho_analysis"].model_dump()["tokens"],
        "phenomena_detected": state.get("phenomena", []),
        "citations": citations,
        "iterations": state.get("iterations", 0),
        "trace_id": str(state["trace_id"]),
    }
    return {"output": output}


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _parse_json(text: str) -> dict[str, Any]:
    """Tollerante: prova a estrarre JSON anche se il modello aggiunge prosa."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: trova il primo { e l'ultimo }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        logger.warning("Impossibile parsare JSON dalla risposta LLM; uso testo grezzo")
        return {"translation": text, "rationale": "", "critique": text, "issues_found": False}


def should_iterate(state: AgentState) -> str:
    """Edge condizionale: se la critique trova problemi e siamo sotto il
    limite di iterazioni, torna a draft_translation. Altrimenti format_output.
    """
    if state.get("issues_found") and state.get("iterations", 0) < MAX_ITERATIONS:
        return "draft_translation"
    return "format_output"
