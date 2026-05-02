"""Unit tests for the Jinja template loader (`agent.prompts.render`).

Verifies that every template using StrictUndefined renders with the
minimal context provided by the LangGraph nodes.
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
                "head": 1,
                "dep_rel": "nsubj",
            },
            {
                "index": 1,
                "text": "est",
                "lemma": "sum",
                "pos": "AUX",
                "features": {"Mood": "Ind"},
                "head": -1,
                "dep_rel": "root",
            },
        ],
        "text": "Gallia est",
    }


def test_render_system_translator() -> None:
    out = render("system_translator.j2")
    assert "philologist" in out.lower()
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
                "target_text": "All Gaul is divided",
                "author": "Caesar",
                "work": "BG",
                "locator": "1.1.1",
                "translator": "McDevitte",
                "distance": 0.123,
            }
        ],
        grammar_hits=[
            {
                "phenomenon": "ablativo_assoluto",
                "description": "Latin participial construction with a noun and a participle, both in the ablative...",
                "examples": [
                    {
                        "lat": "Caesare imperante",
                        "eng": "With Caesar in command",
                        "note": "n/a",
                    }
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
    assert "Previous iteration" not in out


def test_render_draft_translation_with_previous_iteration() -> None:
    out = render(
        "draft_translation.j2",
        input_text="x",
        morpho=_morpho_minimal(),
        phenomena=[],
        tm_hits=[],
        grammar_hits=[],
        previous_draft="previous draft",
        previous_critique="ACI missing",
    )
    assert "Previous iteration" in out
    assert "previous draft" in out
    assert "ACI missing" in out


def test_render_self_critique() -> None:
    out = render(
        "self_critique.j2",
        input_text="Gallia est",
        draft="All Gaul is",
        rationale="reasoning",
        morpho=_morpho_minimal(),
        grammar_hits=[{"phenomenon": "x", "description": "y", "examples": [], "source": None}],
    )
    assert "All Gaul is" in out
    assert "reasoning" in out
