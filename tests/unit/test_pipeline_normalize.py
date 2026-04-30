"""Unit tests for the CLTK→UD feature value normalizer."""

from __future__ import annotations

import pytest

from maggieai.morphology.pipeline import _normalize_ud_value


@pytest.mark.parametrize(
    ("cltk_value", "ud_value"),
    [
        # Case
        ("[ablative]", "Abl"),
        ("[accusative]", "Acc"),
        ("[nominative]", "Nom"),
        # VerbForm
        ("[participle]", "Part"),
        ("[infinitive]", "Inf"),
        ("[gerundive]", "Gdv"),
        # Mood
        ("[indicative]", "Ind"),
        ("[subjunctive]", "Sub"),
        # Voice
        ("[active]", "Act"),
        ("[passive]", "Pass"),
        # Number / Gender
        ("[singular]", "Sing"),
        ("[plural]", "Plur"),
        ("[masculine]", "Masc"),
        # Person — UD uses digits
        ("[third]", "3"),
        # Aspect / Tense
        ("[imperfective]", "Imp"),
        ("[perfective]", "Perf"),
        ("[past]", "Past"),
        ("[pluperfect]", "Pqp"),
    ],
)
def test_normalizes_known_cltk_values(cltk_value: str, ud_value: str) -> None:
    assert _normalize_ud_value(cltk_value) == ud_value


def test_passes_through_multi_value_bundles() -> None:
    """CLTK uses bundles like '[lat_a, nominal]' for InflClass — no UD analogue."""
    assert _normalize_ud_value("[lat_a, nominal]") == "[lat_a, nominal]"


def test_passes_through_unknown_value() -> None:
    """Forward compatibility: an unknown lowercase token is returned as-is."""
    assert _normalize_ud_value("[someThingNew]") == "someThingNew"


def test_ignores_already_ud_value() -> None:
    """Idempotent on an already-stripped UD value."""
    assert _normalize_ud_value("Abl") == "Abl"


def test_handles_bracketless_lowercase() -> None:
    """Defensive: works even when CLTK ever drops the brackets."""
    assert _normalize_ud_value("ablative") == "Abl"
