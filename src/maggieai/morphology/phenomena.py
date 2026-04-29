"""Rilevatore di costrutti sintattici notevoli.

Riceve un `SentenceAnalysis` (output del pipeline CLTK) e applica i
matcher dichiarativi caricati dalla tabella `grammar_rules` per
restituire la lista dei `phenomenon` rilevati.

I matcher sono JSONB con la forma:
    {
      "type": "ud_pattern",
      "match_any": [
        {"upos": "VERB", "Mood": "Sub"},
        {"upos": "NOUN", "Case": "Abl"}
      ]
    }

Per la v1 supportiamo solo `ud_pattern` (match su POS+features).
Pattern più complessi (sequence, dep tree) arriveranno con i casi che
li richiedono — non astraggo prima del bisogno.
"""

from __future__ import annotations

from typing import Any

from maggieai.morphology.pipeline import SentenceAnalysis, TokenAnalysis


def detect(analysis: SentenceAnalysis, rules: list[dict[str, Any]]) -> list[str]:
    """Restituisce i `phenomenon` rilevati dai pattern UD.

    `rules` è la lista di righe `grammar_rules` lette dal DB; ogni elemento
    deve avere almeno `phenomenon` e `pattern`.
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
    # Deduplica preservando l'ordine
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
