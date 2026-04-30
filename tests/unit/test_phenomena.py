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


# Regression guard: ensure the cum_narrativo pattern from
# data/grammar_rules/05_cum_narrativo.yaml fires on the real UD encoding
# of Latin imperfect subjunctive (Aspect=Imp) and pluperfect subjunctive
# (Tense=Pqp). The original pattern used `Tense: Imp` which never
# matches because UD-Latin encodes the imperfect via Aspect, not Tense.
CUM_NARRATIVO = {
    "phenomenon": "cum_narrativo",
    "pattern": {
        "type": "ud_pattern",
        "match_any": [
            {"upos": "VERB", "Mood": "Sub", "Aspect": "Imp"},
            {"upos": "VERB", "Mood": "Sub", "Tense": "Pqp"},
        ],
    },
}


def test_cum_narrativo_matches_imperfect_subjunctive() -> None:
    """`legeret` (imperfect subjunctive) — UD: Mood=Sub, Aspect=Imp, Tense=Past."""
    sent = _sent(
        _tok(0, "legeret", "VERB", Mood="Sub", Aspect="Imp", Tense="Past", VerbForm="Fin"),
    )
    assert detect(sent, [CUM_NARRATIVO]) == ["cum_narrativo"]


def test_cum_narrativo_matches_pluperfect_subjunctive() -> None:
    """`venisset` (pluperfect subjunctive) — UD: Mood=Sub, Aspect=Perf, Tense=Pqp."""
    sent = _sent(
        _tok(0, "venisset", "VERB", Mood="Sub", Aspect="Perf", Tense="Pqp", VerbForm="Fin"),
    )
    assert detect(sent, [CUM_NARRATIVO]) == ["cum_narrativo"]


def test_cum_narrativo_does_not_match_perfect_indicative() -> None:
    """`vicit` (perfect indicative) — UD: Mood=Ind, Aspect=Perf, Tense=Past."""
    sent = _sent(
        _tok(0, "vicit", "VERB", Mood="Ind", Aspect="Perf", Tense="Past", VerbForm="Fin"),
    )
    assert detect(sent, [CUM_NARRATIVO]) == []


def test_cum_narrativo_does_not_match_imperfect_indicative() -> None:
    """`legebat` (imperfect indicative) — UD: Mood=Ind, Aspect=Imp, Tense=Past.

    Same Aspect as the subjunctive imperfect, but Mood=Ind blocks it.
    """
    sent = _sent(
        _tok(0, "legebat", "VERB", Mood="Ind", Aspect="Imp", Tense="Past", VerbForm="Fin"),
    )
    assert detect(sent, [CUM_NARRATIVO]) == []
