"""Unit tests for the pure helpers in `agent.nodes`: `_parse_json`, `should_iterate`."""

from __future__ import annotations

from maggieai.agent.nodes import MAX_ITERATIONS, _parse_json, should_iterate


def test_parse_json_clean() -> None:
    out = _parse_json('{"translation": "hello", "rationale": "x"}')
    assert out == {"translation": "hello", "rationale": "x"}


def test_parse_json_with_prose_around() -> None:
    raw = 'Here is the JSON:\n{"translation": "ok", "rationale": "y"}\n— done.'
    out = _parse_json(raw)
    assert out["translation"] == "ok"
    assert out["rationale"] == "y"


def test_parse_json_invalid_returns_fallback() -> None:
    out = _parse_json("not JSON, no braces either")
    # fallback: wraps the raw text in a dict
    assert "translation" in out
    assert out["issues_found"] is False


def test_parse_json_handles_inner_braces() -> None:
    """Even with prose around and nested braces, the first `{` and last `}` delimit."""
    raw = 'Output: {"a": {"b": 1}, "c": [1,2]}.'
    out = _parse_json(raw)
    assert out == {"a": {"b": 1}, "c": [1, 2]}


def test_should_iterate_loops_when_issues_under_max() -> None:
    state = {"issues_found": True, "iterations": 1}
    assert should_iterate(state) == "draft_translation"  # type: ignore[arg-type]


def test_should_iterate_stops_at_max() -> None:
    state = {"issues_found": True, "iterations": MAX_ITERATIONS}
    assert should_iterate(state) == "format_output"  # type: ignore[arg-type]


def test_should_iterate_stops_when_no_issues() -> None:
    state = {"issues_found": False, "iterations": 0}
    assert should_iterate(state) == "format_output"  # type: ignore[arg-type]


def test_should_iterate_stops_when_keys_missing() -> None:
    assert should_iterate({}) == "format_output"  # type: ignore[arg-type]
