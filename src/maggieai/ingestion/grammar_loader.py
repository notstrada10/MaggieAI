"""Caricatore idempotente delle regole grammaticali da YAML a Postgres.

Ogni file in `data/grammar_rules/*.yaml` rappresenta UN fenomeno
sintattico/morfologico. Schema (vedi `data/grammar_rules/README.md`):

    phenomenon: ablativo_assoluto
    rule_type: syntactic                # 'syntactic' | 'morphological'
    source: "Allen & Greenough §419"
    description: |
      Costrutto participiale latino composto da un sostantivo e un
      participio entrambi all'ablativo, sintatticamente indipendente
      dalla proposizione principale...
    pattern:
      type: ud_pattern
      match_any:
        - { upos: "NOUN", Case: "Abl" }
        - { upos: "VERB", VerbForm: "Part", Case: "Abl" }
    examples:
      - lat: "Caesare imperante, Galli rebellaverunt"
        ita: "Comandando Cesare, i Galli si ribellarono"
        note: "Imperante = participio presente in ablativo"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.dialects.postgresql import insert as pg_insert

from maggieai.db.engine import session_scope
from maggieai.db.models import GrammarRule

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"phenomenon", "rule_type", "description", "pattern"}


def load_directory(directory: Path) -> int:
    """Carica tutti gli `*.yaml` dalla directory in `grammar_rules`.

    Idempotente grazie a `ON CONFLICT (phenomenon, source) DO UPDATE` —
    rieseguire aggiorna le regole esistenti.
    """
    yaml_files = sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml"))
    if not yaml_files:
        logger.warning("Nessun file YAML trovato in %s", directory)
        return 0

    rules: list[dict[str, Any]] = []
    for path in yaml_files:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        _validate(data, path)
        rules.append(_normalize(data))

    with session_scope() as session:
        stmt = pg_insert(GrammarRule).values(rules)
        stmt = stmt.on_conflict_do_update(
            constraint="grammar_rules_phenom_source_uq",
            set_={
                "rule_type": stmt.excluded.rule_type,
                "pattern": stmt.excluded.pattern,
                "description": stmt.excluded.description,
                "examples": stmt.excluded.examples,
            },
        )
        session.execute(stmt)

    logger.info("Caricate %d regole grammaticali da %s", len(rules), directory)
    return len(rules)


def _validate(data: Any, path: Path) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{path}: il file YAML deve contenere un mapping al top-level")
    missing = REQUIRED_FIELDS - data.keys()
    if missing:
        raise ValueError(f"{path}: campi mancanti: {sorted(missing)}")
    if data["rule_type"] not in {"syntactic", "morphological"}:
        raise ValueError(f"{path}: rule_type deve essere 'syntactic' o 'morphological'")


def _normalize(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "phenomenon": data["phenomenon"],
        "rule_type": data["rule_type"],
        "pattern": data["pattern"],
        "description": data["description"].strip(),
        "examples": data.get("examples", []),
        "source": data.get("source"),
    }
