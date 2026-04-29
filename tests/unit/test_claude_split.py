"""Unit tests for `inference.claude_api._split_system`."""

from __future__ import annotations

from maggieai.inference.claude_api import _split_system
from maggieai.inference.client import Message


def test_split_extracts_system_and_keeps_conversation() -> None:
    system, conv = _split_system(
        [
            Message("system", "you are an expert"),
            Message("user", "hi"),
            Message("assistant", "hello"),
        ]
    )
    assert system == "you are an expert"
    assert conv == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_split_concatenates_multiple_system_messages() -> None:
    system, conv = _split_system(
        [
            Message("system", "rule 1"),
            Message("system", "rule 2"),
            Message("user", "x"),
        ]
    )
    assert system == "rule 1\n\nrule 2"
    assert conv == [{"role": "user", "content": "x"}]


def test_split_no_system_returns_none() -> None:
    system, conv = _split_system([Message("user", "hi")])
    assert system is None
    assert conv == [{"role": "user", "content": "hi"}]
