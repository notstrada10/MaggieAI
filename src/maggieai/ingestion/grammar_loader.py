"""Idempotent loader for grammar rules from YAML to Postgres.

Each file in `data/grammar_rules/*.yaml` describes ONE syntactic or
morphological phenomenon. Schema (see `data/grammar_rules/README.md`):

    phenomenon: ablativo_assoluto
    rule_type: syntactic                # 'syntactic' | 'morphological'
    source: "Allen & Greenough §419"
    description: |
      Latin participial construction made up of a noun and a participle
      both in the ablative, syntactically independent of the main
      clause...
    pattern:
      type: ud_pattern
      match_any:
        - { upos: "NOUN", Case: "Abl" }
        - { upos: "VERB", VerbForm: "Part", Case: "Abl" }
    examples:
      - lat: "Caesare imperante, Galli rebellaverunt"
        eng: "With Caesar in command, the Gauls rebelled"
        note: "Imperante = present active participle in the ablative"
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
    """Load every `*.yaml` from `directory` into `grammar_rules`.

    Idempotent thanks to `ON CONFLICT (phenomenon, source) DO UPDATE` —
    re-running updates the existing rules.
    """
    yaml_files = sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml"))
    if not yaml_files:
        logger.warning("No YAML file found in %s", directory)
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

    logger.info("Loaded %d grammar rules from %s", len(rules), directory)
    return len(rules)


def _validate(data: Any, path: Path) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{path}: the YAML file must contain a mapping at top level")
    missing = REQUIRED_FIELDS - data.keys()
    if missing:
        raise ValueError(f"{path}: missing fields: {sorted(missing)}")
    if data["rule_type"] not in {"syntactic", "morphological"}:
        raise ValueError(f"{path}: rule_type must be 'syntactic' or 'morphological'")


def _normalize(data: dict[str, Any]) -> dict[str, Any]:
    description = data["description"].strip()
    template = (data.get("translation_template") or "").strip()
    if template:
        # Surface the template as a structured trailing block so the LLM
        # gets a clear, scoped rendering signal in the grammar_hits
        # section of `draft_translation.j2`. We splice it into the
        # description (rather than adding a new column) to avoid a DB
        # migration for one optional field.
        description = f"{description}\n\n**Render as:** {template}"
    return {
        "phenomenon": data["phenomenon"],
        "rule_type": data["rule_type"],
        "pattern": data["pattern"],
        "description": description,
        "examples": data.get("examples", []),
        "source": data.get("source"),
    }
