"""Unit tests for `maggieai.eval.coverage.compute_coverage`."""

from __future__ import annotations

from maggieai.eval.coverage import Pair, compute_coverage
from maggieai.morphology.pipeline import SentenceAnalysis, TokenAnalysis


def _tok(idx: int, text: str, pos: str, **features: str) -> TokenAnalysis:
    return TokenAnalysis(index=idx, text=text, lemma=text.lower(), pos=pos, features=features)


def _sent(*tokens: TokenAnalysis) -> SentenceAnalysis:
    return SentenceAnalysis(text=" ".join(t.text for t in tokens), tokens=list(tokens))


ABL_ASS = {
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
DEAD = {
    "phenomenon": "never_fires",
    "pattern": {
        "type": "ud_pattern",
        "match_any": [{"upos": "VERB", "Mood": "ImpossibleValue"}],
    },
}


def _pair(pid: int) -> Pair:
    return Pair(id=pid, source_text=f"text-{pid}", author=None, work=None, locator=None)


def test_counts_hits_per_rule() -> None:
    items = [
        (_pair(1), _sent(_tok(0, "imperante", "VERB", VerbForm="Part", Case="Abl"))),
        (_pair(2), _sent(_tok(0, "esse", "VERB", VerbForm="Inf"))),
        (_pair(3), _sent(_tok(0, "imperante", "VERB", VerbForm="Part", Case="Abl"))),
    ]
    report = compute_coverage(items, [ABL_ASS, ACI])
    assert report.counts == {"ablativo_assoluto": 2, "accusativo_con_infinito": 1}
    assert report.pairs_analyzed == 3
    assert report.errors == 0


def test_dead_rule_listed_but_not_in_counts() -> None:
    items = [(_pair(1), _sent(_tok(0, "imperante", "VERB", VerbForm="Part", Case="Abl")))]
    report = compute_coverage(items, [ABL_ASS, DEAD])
    assert "never_fires" in report.rules_seen
    assert "never_fires" not in report.counts
    assert report.counts == {"ablativo_assoluto": 1}


def test_failed_analyses_increment_errors() -> None:
    items = [
        (_pair(1), None),
        (_pair(2), _sent(_tok(0, "imperante", "VERB", VerbForm="Part", Case="Abl"))),
        (_pair(3), None),
    ]
    report = compute_coverage(items, [ABL_ASS])
    assert report.errors == 2
    assert report.pairs_analyzed == 1
    assert report.counts == {"ablativo_assoluto": 1}


def test_sample_hits_capped() -> None:
    items = [
        (_pair(i), _sent(_tok(0, "imperante", "VERB", VerbForm="Part", Case="Abl")))
        for i in range(1, 11)
    ]
    report = compute_coverage(items, [ABL_ASS], sample_per_rule=3)
    assert report.sample_hits["ablativo_assoluto"] == [1, 2, 3]


def test_one_pair_one_count_even_with_repeated_match() -> None:
    """The detector dedupes per-sentence; coverage inherits that."""
    items = [
        (
            _pair(1),
            _sent(
                _tok(0, "imperante", "VERB", VerbForm="Part", Case="Abl"),
                _tok(1, "expulso", "VERB", VerbForm="Part", Case="Abl"),
            ),
        )
    ]
    report = compute_coverage(items, [ABL_ASS])
    assert report.counts == {"ablativo_assoluto": 1}


def test_empty_items_yields_zero_report() -> None:
    report = compute_coverage([], [ABL_ASS])
    assert report.counts == {}
    assert report.errors == 0
    assert report.pairs_analyzed == 0
    assert report.rules_seen == ["ablativo_assoluto"]
