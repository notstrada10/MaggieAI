"""Unit tests for `grammar_loader` validation/normalization (no DB)."""

from __future__ import annotations

from pathlib import Path

import pytest

from maggieai.ingestion.grammar_loader import _normalize, _validate


def _ok_data() -> dict[str, object]:
    return {
        "phenomenon": "ablativo_assoluto",
        "rule_type": "syntactic",
        "description": "  A description.\n",
        "pattern": {"type": "ud_pattern", "match_any": []},
        "source": "A&G §419",
        "examples": [{"lat": "x", "eng": "y"}],
    }


def test_validate_accepts_complete() -> None:
    _validate(_ok_data(), Path("dummy.yaml"))


def test_validate_rejects_missing_required() -> None:
    bad = _ok_data()
    del bad["pattern"]
    with pytest.raises(ValueError, match="missing fields"):
        _validate(bad, Path("dummy.yaml"))


def test_validate_rejects_bad_rule_type() -> None:
    bad = _ok_data()
    bad["rule_type"] = "semantic"
    with pytest.raises(ValueError, match="rule_type"):
        _validate(bad, Path("dummy.yaml"))


def test_validate_rejects_non_dict() -> None:
    with pytest.raises(ValueError, match="mapping"):
        _validate(["not", "a", "dict"], Path("dummy.yaml"))


def test_normalize_strips_description_and_defaults_examples() -> None:
    data = _ok_data()
    data["description"] = "   text with whitespace   \n"
    data.pop("examples")
    out = _normalize(data)
    assert out["description"] == "text with whitespace"
    assert out["examples"] == []
    assert out["source"] == "A&G §419"


def test_normalize_passes_pattern_through() -> None:
    out = _normalize(_ok_data())
    assert out["pattern"] == {"type": "ud_pattern", "match_any": []}


def test_load_real_grammar_files_pass_validation() -> None:
    """Smoke check: every versioned YAML in data/grammar_rules passes
    validation and normalises into a usable row.
    """
    import yaml

    grammar_dir = Path(__file__).resolve().parents[2] / "data" / "grammar_rules"
    yaml_files = sorted(grammar_dir.glob("*.yaml"))
    assert len(yaml_files) >= 12, f"expected at least 12 files, found {len(yaml_files)}"
    seen_phenomena: set[str] = set()
    for path in yaml_files:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        _validate(data, path)
        out = _normalize(data)
        assert out["phenomenon"]
        assert out["pattern"]
        assert out["phenomenon"] not in seen_phenomena, (
            f"duplicate phenomenon slug: {out['phenomenon']} in {path.name}"
        )
        seen_phenomena.add(out["phenomenon"])
