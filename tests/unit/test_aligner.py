"""Unit tests for `ingestion.aligner.align_by_locator`."""

from __future__ import annotations

from maggieai.ingestion.aligner import TranslationSegment, align_by_locator
from maggieai.ingestion.perseus import LatinSegment


def _lat(book: int, chapter: int, section: int | None, text: str) -> LatinSegment:
    return LatinSegment(text=text, book=book, chapter=chapter, section=section)


def test_align_full_match() -> None:
    latins = [
        _lat(1, 1, 1, "Gallia est omnis divisa"),
        _lat(1, 1, 2, "quarum unam incolunt Belgae"),
    ]
    translations = [
        TranslationSegment(text="All Gaul is divided", locator="1.1.1", translator="X"),
        TranslationSegment(text="one of which the Belgae inhabit", locator="1.1.2", translator="X"),
    ]
    pairs = align_by_locator(latins, translations)
    assert len(pairs) == 2
    assert pairs[0].locator == "1.1.1"
    assert pairs[0].source_text == "Gallia est omnis divisa"
    assert pairs[0].target_text == "All Gaul is divided"
    assert pairs[1].locator == "1.1.2"


def test_align_drops_unmatched_latin() -> None:
    latins = [
        _lat(1, 1, 1, "A"),
        _lat(1, 1, 2, "B"),  # no translation match
    ]
    translations = [
        TranslationSegment(text="alpha", locator="1.1.1", translator="X"),
    ]
    pairs = align_by_locator(latins, translations)
    assert len(pairs) == 1
    assert pairs[0].source_text == "A"


def test_align_ignores_extra_translation() -> None:
    latins = [_lat(2, 3, None, "qux")]
    translations = [
        TranslationSegment(text="qux-en", locator="2.3", translator="X"),
        TranslationSegment(text="orphan", locator="9.9.9", translator="X"),
    ]
    pairs = align_by_locator(latins, translations)
    assert len(pairs) == 1
    assert pairs[0].locator == "2.3"


def test_align_empty_inputs() -> None:
    assert align_by_locator([], []) == []
