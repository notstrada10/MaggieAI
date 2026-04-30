"""Unit tests for `ingestion.perseus.parse_dbg`.

Uses a synthetic mini-TEI XML to avoid network dependency.
"""

from __future__ import annotations

from maggieai.ingestion.perseus import parse_dbg

TEI = b"""<?xml version="1.0"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text>
    <body>
      <div type="edition">
        <div type="textpart" subtype="book" n="1">
          <div type="textpart" subtype="chapter" n="1">
            <div type="textpart" subtype="section" n="1">
              <p>Gallia est omnis divisa in partes tres.</p>
            </div>
            <div type="textpart" subtype="section" n="2">
              <p>quarum unam incolunt Belgae.</p>
            </div>
          </div>
          <div type="textpart" subtype="chapter" n="2">
            <p>chapter without sections</p>
          </div>
        </div>
        <div type="textpart" subtype="book" n="2">
          <div type="textpart" subtype="chapter" n="1">
            <div type="textpart" subtype="section" n="1">
              <p>second book</p>
            </div>
          </div>
        </div>
      </div>
    </body>
  </text>
</TEI>
"""


def test_parse_returns_segments_with_locator() -> None:
    segs = parse_dbg(TEI)
    locators = {s.locator for s in segs}
    assert "1.1.1" in locators
    assert "1.1.2" in locators
    assert "1.2" in locators  # chapter without sections
    assert "2.1.1" in locators


def test_parse_filters_by_book() -> None:
    segs = parse_dbg(TEI, books=[1])
    books = {s.book for s in segs}
    assert books == {1}


def test_parse_extracts_clean_text() -> None:
    segs = parse_dbg(TEI, books=[1])
    by_loc = {s.locator: s.text for s in segs}
    assert by_loc["1.1.1"] == "Gallia est omnis divisa in partes tres."
    assert by_loc["1.2"] == "chapter without sections"


def test_parse_chapter_without_sections_has_no_section() -> None:
    segs = parse_dbg(TEI, books=[1])
    chap2 = next(s for s in segs if s.book == 1 and s.chapter == 2)
    assert chap2.section is None


def test_parse_chapter_granularity_collapses_sections() -> None:
    """granularity='chapter' aggregates section text into the chapter."""
    segs = parse_dbg(TEI, books=[1], granularity="chapter")
    locators = {s.locator for s in segs}
    # No section-level locators emitted
    assert "1.1.1" not in locators
    assert "1.1.2" not in locators
    # Chapter-level locators present (both 1.1 and 1.2)
    assert "1.1" in locators
    assert "1.2" in locators
    # Concatenated text contains both sections from chapter 1.1
    chap_1_1 = next(s for s in segs if s.locator == "1.1")
    assert "Gallia est omnis divisa" in chap_1_1.text
    assert "quarum unam incolunt Belgae" in chap_1_1.text
    # Section field is None at chapter granularity
    assert chap_1_1.section is None
