"""Detector for notable syntactic constructs.

Receives a `SentenceAnalysis` (output of the CLTK pipeline) and applies
the declarative matchers loaded from the `grammar_rules` table to
return the list of detected `phenomenon` slugs.

Matchers are JSONB of the shape:
    {
      "type": "ud_pattern",
      "match_any": [
        {"upos": "VERB", "Mood": "Sub"},
        {"upos": "NOUN", "Case": "Abl"}
      ]
    }

For v1 we only support `ud_pattern` (match on POS+features). More
complex patterns (sequence, dep tree) will arrive when a real case
requires them — no preemptive abstraction.
"""

from __future__ import annotations

from typing import Any

from maggieai.morphology.pipeline import SentenceAnalysis, TokenAnalysis


def detect(analysis: SentenceAnalysis, rules: list[dict[str, Any]]) -> list[str]:
    """Return the `phenomenon` slugs detected by the UD patterns.

    `rules` is the list of `grammar_rules` rows read from the DB; each
    element must have at least `phenomenon` and `pattern`.
    """
    found: list[str] = []
    for rule in rules:
        pattern = rule.get("pattern", {})
        if pattern.get("type") != "ud_pattern":
            continue
        match_any = pattern.get("match_any", [])
        for token in analysis.tokens:
            if any(_token_matches(token, m) for m in match_any):
                found.append(rule["phenomenon"])
                break
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for p in found:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _token_matches(token: TokenAnalysis, criteria: dict[str, Any]) -> bool:
    for key, expected in criteria.items():
        if key == "upos":
            if token.pos != expected:
                return False
        else:
            if token.features.get(key) != expected:
                return False
    return True
