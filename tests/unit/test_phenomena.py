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


# Regression guards for the gerundive (perifrastica_passiva). The CLTK
# Stanza ittb model encodes the gerundive as VerbForm=Part+Aspect=prospective
# rather than the canonical UD VerbForm=Gdv — see comment in
# data/grammar_rules/06_perifrastica_passiva.yaml.
PERIFRASTICA_PASSIVA = {
    "phenomenon": "perifrastica_passiva",
    "pattern": {
        "type": "ud_pattern",
        "match_any": [
            {"upos": "VERB", "VerbForm": "Gdv"},
            {"upos": "ADJ", "VerbForm": "Gdv"},
            {"upos": "VERB", "VerbForm": "Part", "Aspect": "prospective"},
        ],
    },
}


def test_perifrastica_passiva_matches_cltk_prospective_aspect() -> None:
    """`delenda` (gerundive) as CLTK Stanza ittb tags it."""
    sent = _sent(
        _tok(0, "delenda", "VERB", VerbForm="Part", Aspect="prospective", Voice="Pass"),
    )
    assert detect(sent, [PERIFRASTICA_PASSIVA]) == ["perifrastica_passiva"]


def test_perifrastica_passiva_matches_canonical_ud_gdv() -> None:
    """Forward compat with treebanks that follow canonical UD-Latin."""
    sent = _sent(_tok(0, "delenda", "VERB", VerbForm="Gdv"))
    assert detect(sent, [PERIFRASTICA_PASSIVA]) == ["perifrastica_passiva"]


def test_perifrastica_passiva_does_not_match_perfect_passive_participle() -> None:
    """Control: regular PPP (`expulso`) has Aspect=Perf, NOT prospective."""
    sent = _sent(
        _tok(0, "expulso", "VERB", VerbForm="Part", Aspect="Perf", Voice="Pass"),
    )
    assert detect(sent, [PERIFRASTICA_PASSIVA]) == []


# Regression guards for perifrastica_attiva. CLTK Stanza ittb tagging is
# inconsistent on future active participles — see YAML for full notes.
PERIFRASTICA_ATTIVA = {
    "phenomenon": "perifrastica_attiva",
    "pattern": {
        "type": "ud_pattern",
        "match_any": [
            {"upos": "VERB", "VerbForm": "Part", "Aspect": "Perf", "Voice": "Act"},
        ],
    },
}


def test_perifrastica_attiva_matches_correctly_tagged_morituri() -> None:
    """`Morituri` came back from CLTK as Aspect=Perf, Voice=Act — pattern catches it."""
    sent = _sent(
        _tok(0, "Morituri", "VERB", VerbForm="Part", Aspect="Perf", Voice="Act"),
    )
    assert detect(sent, [PERIFRASTICA_ATTIVA]) == ["perifrastica_attiva"]


def test_perifrastica_attiva_misses_mistagged_pugnaturus() -> None:
    """Document the known model wart: `pugnaturus` is mistagged as Voice=Pass.

    The pattern can't catch it without over-matching all PPPs. The LLM
    falls back to the morpho table to identify it.
    """
    sent = _sent(
        _tok(0, "pugnaturus", "VERB", VerbForm="Part", Aspect="Perf", Voice="Pass"),
    )
    assert detect(sent, [PERIFRASTICA_ATTIVA]) == []


# Pattern-key extensions: lemma, lemma_in, dep_rel
def _tok_lemma(idx: int, text: str, pos: str, lemma: str, **features: str) -> TokenAnalysis:
    return TokenAnalysis(index=idx, text=text, lemma=lemma, pos=pos, features=features)


def _tok_dep(idx: int, text: str, pos: str, dep_rel: str, **features: str) -> TokenAnalysis:
    return TokenAnalysis(
        index=idx, text=text, lemma=text.lower(), pos=pos, features=features, dep_rel=dep_rel
    )


CUM_INDICATIVE = {
    "phenomenon": "cum_indicativo",
    "pattern": {
        "type": "ud_pattern",
        "match_any": [{"upos": "SCONJ", "lemma": "cum"}],
    },
}

POSTQUAM = {
    "phenomenon": "postquam_ubi_simul",
    "pattern": {
        "type": "ud_pattern",
        "match_any": [{"upos": "SCONJ", "lemma_in": ["postquam", "ubi", "simul"]}],
    },
}

REL_CHARACTERISTIC = {
    "phenomenon": "relativa_caratteristica",
    "pattern": {
        "type": "ud_pattern",
        "match_any": [{"upos": "VERB", "Mood": "Sub", "dep_rel": "acl:relcl"}],
    },
}


def test_lemma_match_cum() -> None:
    sent = _sent(_tok_lemma(0, "cum", "SCONJ", "cum"))
    assert detect(sent, [CUM_INDICATIVE]) == ["cum_indicativo"]


def test_lemma_match_is_case_insensitive() -> None:
    """CLTK occasionally returns capitalised lemmas; the matcher must
    normalise both sides."""
    sent = _sent(_tok_lemma(0, "Cum", "SCONJ", "Cum"))
    assert detect(sent, [CUM_INDICATIVE]) == ["cum_indicativo"]


def test_lemma_does_not_match_different_word() -> None:
    sent = _sent(_tok_lemma(0, "ut", "SCONJ", "ut"))
    assert detect(sent, [CUM_INDICATIVE]) == []


def test_lemma_in_matches_any_listed() -> None:
    sent = _sent(_tok_lemma(0, "Postquam", "SCONJ", "postquam"))
    assert detect(sent, [POSTQUAM]) == ["postquam_ubi_simul"]
    sent2 = _sent(_tok_lemma(0, "ubi", "SCONJ", "ubi"))
    assert detect(sent2, [POSTQUAM]) == ["postquam_ubi_simul"]


def test_lemma_in_does_not_match_unlisted() -> None:
    sent = _sent(_tok_lemma(0, "antequam", "SCONJ", "antequam"))
    assert detect(sent, [POSTQUAM]) == []


def test_dep_rel_match() -> None:
    sent = _sent(_tok_dep(0, "amet", "VERB", "acl:relcl", Mood="Sub"))
    assert detect(sent, [REL_CHARACTERISTIC]) == ["relativa_caratteristica"]


def test_dep_rel_does_not_match_when_relation_differs() -> None:
    sent = _sent(_tok_dep(0, "amet", "VERB", "advcl", Mood="Sub"))
    assert detect(sent, [REL_CHARACTERISTIC]) == []
