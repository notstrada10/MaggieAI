"""Unit tests for the Jinja template loader (`agent.prompts.render`).

Verifies that every template using StrictUndefined renders with the
minimal context provided by the LangGraph nodes.

NOTE: Some assertions still reference Italian content of the prompts
(e.g. "filologo", "Iterazione precedente"). Those will be flipped to
English in Phase B when the .j2 files themselves are translated.
"""

from __future__ import annotations

from maggieai.agent.prompts import render


def _morpho_minimal() -> dict[str, object]:
    return {
        "tokens": [
            {
                "index": 0,
                "text": "Gallia",
                "lemma": "Gallia",
                "pos": "PROPN",
                "features": {"Case": "Nom", "Number": "Sing"},
            },
            {
                "index": 1,
                "text": "est",
                "lemma": "sum",
                "pos": "AUX",
                "features": {"Mood": "Ind"},
            },
        ],
        "text": "Gallia est",
    }


def test_render_system_translator() -> None:
    out = render("system_translator.j2")
    assert "filologo" in out.lower()
    assert "JSON" in out


def test_render_system_critic() -> None:
    out = render("system_critic.j2")
    assert "issues_found" in out


def test_render_draft_translation_full_context() -> None:
    out = render(
        "draft_translation.j2",
        input_text="Gallia est",
        morpho=_morpho_minimal(),
        phenomena=["ablativo_assoluto"],
        tm_hits=[
            {
                "source_text": "Gallia est omnis",
                "target_text": "La Gallia è tutta",
                "author": "Caesar",
                "work": "BG",
                "locator": "1.1.1",
                "translator": "Cocci",
                "distance": 0.123,
            }
        ],
        grammar_hits=[
            {
                "phenomenon": "ablativo_assoluto",
                "description": "Costrutto participiale...",
                "examples": [
                    {"lat": "Caesare imperante", "ita": "Comandando Cesare", "note": "n/a"}
                ],
                "source": "A&G §419",
            }
        ],
        previous_draft=None,
        previous_critique=None,
    )
    assert "Gallia est" in out
    assert "ablativo_assoluto" in out
    assert "BG" in out
    assert "Iterazione precedente" not in out


def test_render_draft_translation_with_previous_iteration() -> None:
    out = render(
        "draft_translation.j2",
        input_text="x",
        morpho=_morpho_minimal(),
        phenomena=[],
        tm_hits=[],
        grammar_hits=[],
        previous_draft="bozza precedente",
        previous_critique="manca AcI",
    )
    assert "Iterazione precedente" in out
    assert "bozza precedente" in out
    assert "manca AcI" in out


def test_render_self_critique() -> None:
    out = render(
        "self_critique.j2",
        input_text="Gallia est",
        draft="La Gallia è",
        rationale="motivazione",
        morpho=_morpho_minimal(),
        grammar_hits=[{"phenomenon": "x", "description": "y", "examples": [], "source": None}],
    )
    assert "La Gallia è" in out
    assert "motivazione" in out
