"""Unit tests for `morphology.phenomena.detect` (UD matcher)."""

from __future__ import annotations

from maggieai.morphology.phenomena import detect
from maggieai.morphology.pipeline import SentenceAnalysis, TokenAnalysis


def _tok(idx: int, text: str, pos: str, **features: str) -> TokenAnalysis:
    return TokenAnalysis(index=idx, text=text, lemma=text.lower(), pos=pos, features=features)


def _sent(*tokens: TokenAnalysis) -> SentenceAnalysis:
    return SentenceAnalysis(text=" ".join(t.text for t in tokens), tokens=list(tokens))


ABL_ASSOLUTO = {
    "phenomenon": "ablativo_assoluto",
    "pattern": {
        "type": "ud_pattern",
        "match_any": [{"upos": "VERB", "VerbForm": "Part", "Case": "Abl"}],
    },
}

ACI = {
    "phenomenon": "accusativo_con_infinito",
    "pattern": {
        "type": "ud_pattern",
        "match_any": [{"upos": "VERB", "VerbForm": "Inf"}],
    },
}


def test_detects_ablativo_assoluto() -> None:
    sent = _sent(
        _tok(0, "Caesare", "PROPN", Case="Abl"),
        _tok(1, "imperante", "VERB", VerbForm="Part", Case="Abl"),
    )
    assert detect(sent, [ABL_ASSOLUTO]) == ["ablativo_assoluto"]


def test_no_match_when_features_differ() -> None:
    sent = _sent(_tok(0, "Caesar", "PROPN", Case="Nom"))
    assert detect(sent, [ABL_ASSOLUTO]) == []


def test_dedupes_repeated_phenomena() -> None:
    """Two tokens that trigger the same phenomenon → only one entry."""
    sent = _sent(
        _tok(0, "imperante", "VERB", VerbForm="Part", Case="Abl"),
        _tok(1, "expulso", "VERB", VerbForm="Part", Case="Abl"),
    )
    assert detect(sent, [ABL_ASSOLUTO]) == ["ablativo_assoluto"]


def test_detects_multiple_phenomena_in_order() -> None:
    sent = _sent(
        _tok(0, "imperante", "VERB", VerbForm="Part", Case="Abl"),
        _tok(1, "esse", "VERB", VerbForm="Inf"),
    )
    out = detect(sent, [ABL_ASSOLUTO, ACI])
    assert set(out) == {"ablativo_assoluto", "accusativo_con_infinito"}


def test_ignores_unknown_pattern_type() -> None:
    sent = _sent(_tok(0, "imperante", "VERB", VerbForm="Part", Case="Abl"))
    bogus = {"phenomenon": "x", "pattern": {"type": "magic"}}
    assert detect(sent, [bogus]) == []


def test_empty_rules_returns_empty() -> None:
    sent = _sent(_tok(0, "imperante", "VERB", VerbForm="Part", Case="Abl"))
    assert detect(sent, []) == []
