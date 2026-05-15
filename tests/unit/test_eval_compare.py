"""Pure helpers behind `maggie-eval compare`: load, align, score."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from maggieai.eval.runner import (
    _align_runs,
    _index_run_by_key,
    _load_run,
    _needs_rerun,
    _record_key,
    _sentence_bleu,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_load_run_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    p.write_text(
        '{"input": "a", "reference": "A", "hypothesis": "a", "error": null}\n'
        "\n"
        '{"input": "b", "reference": "B", "hypothesis": "b", "error": null}\n',
        encoding="utf-8",
    )
    assert len(_load_run(p)) == 2


def test_align_runs_by_question_id(tmp_path: Path) -> None:
    a = [
        {"question_id": "q1", "input": "x", "reference": "R", "hypothesis": "ha", "error": None},
        {"question_id": "q2", "input": "y", "reference": "R", "hypothesis": "hb", "error": None},
    ]
    b = [
        # Order shuffled and one item missing
        {"question_id": "q2", "input": "y", "reference": "R", "hypothesis": "hc", "error": None},
        {"question_id": "q3", "input": "z", "reference": "R", "hypothesis": "hd", "error": None},
    ]
    aligned, warnings = _align_runs(a, b)
    assert [pair[0]["question_id"] for pair in aligned] == ["q2"]
    # Items only in one side produce a warning each
    assert any("only in A" in w for w in warnings)
    assert any("only in B" in w for w in warnings)


def test_align_runs_falls_back_to_input_when_no_qid() -> None:
    """Some legacy outputs may lack question_id — keyed by input then."""
    a = [{"input": "Caesare imperante", "reference": "...", "hypothesis": "x", "error": None}]
    b = [{"input": "Caesare imperante", "reference": "...", "hypothesis": "y", "error": None}]
    aligned, warnings = _align_runs(a, b)
    assert len(aligned) == 1
    assert warnings == []


def test_sentence_bleu_handles_error_case_as_zero() -> None:
    """Empty hypothesis (the error case) should score zero, not crash sacrebleu."""
    assert _sentence_bleu("", "Some reference text.") == 0.0


def test_sentence_bleu_perfect_match_is_100() -> None:
    # sacrebleu's geometric mean rounds to ~100.0000000000004 on a perfect match
    assert _sentence_bleu("Hello world.", "Hello world.") == pytest.approx(100.0)


# --- resume-from helpers --------------------------------------------------


def test_record_key_prefers_question_id_over_input() -> None:
    assert _record_key({"question_id": "q1", "input": "x"}) == "q1"
    assert _record_key({"input": "fallback"}) == "fallback"


def test_needs_rerun_flags_errors() -> None:
    assert _needs_rerun({"hypothesis": "ok", "error": "ConnectError: dns"}) is True


def test_needs_rerun_flags_empty_hypothesis() -> None:
    # Empty / whitespace-only hyp also gets re-run even if `error` is None.
    assert _needs_rerun({"hypothesis": "", "error": None}) is True
    assert _needs_rerun({"hypothesis": "   ", "error": None}) is True


def test_needs_rerun_keeps_good_rows() -> None:
    assert _needs_rerun({"hypothesis": "Caesar ruled.", "error": None}) is False


def test_index_run_by_key_uses_question_id_then_input() -> None:
    items = [
        {"question_id": "q1", "input": "a", "hypothesis": "ha"},
        {"input": "b", "hypothesis": "hb"},  # no qid → keyed by input
    ]
    idx = _index_run_by_key(items)
    assert idx["q1"]["hypothesis"] == "ha"
    assert idx["b"]["hypothesis"] == "hb"
