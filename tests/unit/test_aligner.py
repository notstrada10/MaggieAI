"""Unit test per `ingestion.aligner.align_by_locator`."""

from __future__ import annotations

from maggieai.ingestion.aligner import ItalianSegment, align_by_locator
from maggieai.ingestion.perseus import LatinSegment


def _lat(book: int, chapter: int, section: int | None, text: str) -> LatinSegment:
    return LatinSegment(text=text, book=book, chapter=chapter, section=section)


def test_align_full_match() -> None:
    latins = [
        _lat(1, 1, 1, "Gallia est omnis divisa"),
        _lat(1, 1, 2, "quarum unam incolunt Belgae"),
    ]
    italians = [
        ItalianSegment(text="La Gallia è tutta divisa", locator="1.1.1", translator="X"),
        ItalianSegment(text="una delle quali la abitano i Belgi", locator="1.1.2", translator="X"),
    ]
    pairs = align_by_locator(latins, italians)
    assert len(pairs) == 2
    assert pairs[0].locator == "1.1.1"
    assert pairs[0].source_text == "Gallia est omnis divisa"
    assert pairs[0].target_text == "La Gallia è tutta divisa"
    assert pairs[1].locator == "1.1.2"


def test_align_drops_unmatched_latin() -> None:
    latins = [
        _lat(1, 1, 1, "A"),
        _lat(1, 1, 2, "B"),  # nessun match italiano
    ]
    italians = [
        ItalianSegment(text="alfa", locator="1.1.1", translator="X"),
    ]
    pairs = align_by_locator(latins, italians)
    assert len(pairs) == 1
    assert pairs[0].source_text == "A"


def test_align_ignores_extra_italian() -> None:
    latins = [_lat(2, 3, None, "qux")]
    italians = [
        ItalianSegment(text="qux-it", locator="2.3", translator="X"),
        ItalianSegment(text="orphan", locator="9.9.9", translator="X"),
    ]
    pairs = align_by_locator(latins, italians)
    assert len(pairs) == 1
    assert pairs[0].locator == "2.3"


def test_align_empty_inputs() -> None:
    assert align_by_locator([], []) == []
