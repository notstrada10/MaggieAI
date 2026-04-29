"""Unit test per `inference.claude_api._split_system`."""

from __future__ import annotations

from maggieai.inference.claude_api import _split_system
from maggieai.inference.client import Message


def test_split_extracts_system_and_keeps_conversation() -> None:
    system, conv = _split_system(
        [
            Message("system", "sei un esperto"),
            Message("user", "ciao"),
            Message("assistant", "salve"),
        ]
    )
    assert system == "sei un esperto"
    assert conv == [{"role": "user", "content": "ciao"}, {"role": "assistant", "content": "salve"}]


def test_split_concatenates_multiple_system_messages() -> None:
    system, conv = _split_system(
        [
            Message("system", "regola 1"),
            Message("system", "regola 2"),
            Message("user", "x"),
        ]
    )
    assert system == "regola 1\n\nregola 2"
    assert conv == [{"role": "user", "content": "x"}]


def test_split_no_system_returns_none() -> None:
    system, conv = _split_system([Message("user", "ciao")])
    assert system is None
    assert conv == [{"role": "user", "content": "ciao"}]
