"""Scraper for the Perseus Digital Library (Caesar — De Bello Gallico).

Source: the `PerseusDL/canonical-latinLit` GitHub repo exposes both
the canonical Latin text AND public-domain English translations in
TEI XML (license CC BY-SA 4.0). Same schema for both — only the
`-lat2` vs `-eng2` filename suffix changes.

URLs:
- Latin: https://raw.githubusercontent.com/PerseusDL/canonical-latinLit/master/data/phi0448/phi001/phi0448.phi001.perseus-lat2.xml
- English: https://raw.githubusercontent.com/PerseusDL/canonical-latinLit/master/data/phi0448/phi001/phi0448.phi001.perseus-eng2.xml
  (McDevitte & Bohn 1869 translation, re-licensed by Perseus as CC BY-SA)

Granularity note: the Latin XML provides section-level divs
(`book.chapter.section` locators); the English XML stops at chapter
(`book.chapter`). The `granularity` parameter on :func:`parse_dbg`
controls which level the parser emits, allowing the two sides to be
aligned at the coarser common level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import requests
from lxml import etree

logger = logging.getLogger(__name__)

_PERSEUS_BASE = (
    "https://raw.githubusercontent.com/PerseusDL/canonical-latinLit/master/data"
)

DBG_LATIN_URL = f"{_PERSEUS_BASE}/phi0448/phi001/phi0448.phi001.perseus-lat2.xml"
DBG_ENG_URL = f"{_PERSEUS_BASE}/phi0448/phi001/phi0448.phi001.perseus-eng2.xml"
DBC_LATIN_URL = f"{_PERSEUS_BASE}/phi0448/phi002/phi0448.phi002.perseus-lat2.xml"
DBC_ENG_URL = f"{_PERSEUS_BASE}/phi0448/phi002/phi0448.phi002.perseus-eng2.xml"

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

Granularity = Literal["section", "chapter"]


@dataclass(frozen=True)
class PerseusWork:
    """One author+work entry in the Perseus catalog.

    Adding a new corpus = appending a new entry to :data:`PERSEUS_WORKS`.
    Both URLs must point to TEI XML files sharing the same canonical
    book.chapter[.section] hierarchy used by :func:`parse_dbg`.
    """

    slug: str
    author: str
    work: str
    lat_url: str
    eng_url: str
    eng_translator: str
    license: str = "CC BY-SA 4.0 (Perseus)"


PERSEUS_WORKS: dict[str, PerseusWork] = {
    "dbg": PerseusWork(
        slug="dbg",
        author="Caesar",
        work="De Bello Gallico",
        lat_url=DBG_LATIN_URL,
        eng_url=DBG_ENG_URL,
        eng_translator="McDevitte & Bohn (1869)",
    ),
    "dbc": PerseusWork(
        slug="dbc",
        author="Caesar",
        work="De Bello Civili",
        lat_url=DBC_LATIN_URL,
        eng_url=DBC_ENG_URL,
        eng_translator="Peskett (1914)",
    ),
}


@dataclass(frozen=True)
class LatinSegment:
    text: str
    book: int
    chapter: int
    section: int | None  # some paragraphs are not divided into sections

    @property
    def locator(self) -> str:
        if self.section is None:
            return f"{self.book}.{self.chapter}"
        return f"{self.book}.{self.chapter}.{self.section}"


def fetch_dbg_xml(url: str = DBG_LATIN_URL, timeout: float = 30.0) -> bytes:
    logger.info("Fetching De Bello Gallico TEI XML from Perseus")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def parse_dbg(
    xml_bytes: bytes,
    books: list[int] | None = None,
    granularity: Granularity = "section",
) -> list[LatinSegment]:
    """Parse the TEI XML and return segments, optionally filtered by book.

    The Perseus TEI structure for the DBG is:

        <div type="edition">
          <div type="textpart" subtype="book" n="1">
            <div type="textpart" subtype="chapter" n="1">
              <div type="textpart" subtype="section" n="1">
                <p>Gallia est omnis...</p>
              </div>
              ...

    `granularity` selects the level emitted in `LatinSegment.section`:
    - ``"section"`` (default): one segment per section when sections exist;
      one segment per chapter for chapters that lack sections (legacy
      behaviour, used for Latin where the canonical hierarchy is fully
      populated).
    - ``"chapter"``: one segment per chapter regardless of whether
      sections exist; section-level content is concatenated within the
      chapter. Used to align with translations (e.g. the Perseus English
      file) that stop at chapter granularity.
    """
    root = etree.fromstring(xml_bytes)
    segments: list[LatinSegment] = []

    book_divs = root.xpath(
        "//tei:div[@type='textpart' and @subtype='book']",
        namespaces=TEI_NS,
    )
    for book_div in book_divs:
        book_n = _safe_int(book_div.get("n"))
        if book_n is None or (books is not None and book_n not in books):
            continue
        chapter_divs = book_div.xpath(
            ".//tei:div[@type='textpart' and @subtype='chapter']",
            namespaces=TEI_NS,
        )
        for chap_div in chapter_divs:
            chap_n = _safe_int(chap_div.get("n"))
            if chap_n is None:
                continue
            section_divs = chap_div.xpath(
                ".//tei:div[@type='textpart' and @subtype='section']",
                namespaces=TEI_NS,
            )
            if granularity == "section" and section_divs:
                for sec_div in section_divs:
                    sec_n = _safe_int(sec_div.get("n"))
                    text = _collect_text(sec_div)
                    if text:
                        segments.append(
                            LatinSegment(text=text, book=book_n, chapter=chap_n, section=sec_n)
                        )
            else:
                # Either granularity=="chapter" (always emit chapter-level)
                # or no <section> divs are present in this chapter.
                text = _collect_text(chap_div)
                if text:
                    segments.append(
                        LatinSegment(text=text, book=book_n, chapter=chap_n, section=None)
                    )

    logger.info(
        "Parsing complete: %d segments extracted (granularity=%s)", len(segments), granularity
    )
    return segments


def _collect_text(elem: etree._Element) -> str:
    """Concatenate all child <p> text, normalize whitespace."""
    paragraphs = elem.xpath(".//tei:p", namespaces=TEI_NS)
    chunks: list[str] = []
    for p in paragraphs:
        text = "".join(p.itertext()).strip()
        if text:
            chunks.append(text)
    joined = " ".join(chunks)
    return " ".join(joined.split())


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
