"""Scraper per la Perseus Digital Library (Caesar — De Bello Gallico).

Sorgente: il repo `PerseusDL/canonical-latinLit` su GitHub espone testi
latini canonici in TEI XML (license CC BY-SA 3.0 USA). È più robusto
dello scraping HTML del sito Perseus.

Per la v1 scarichiamo solo il *De Bello Gallico* di Cesare:
https://raw.githubusercontent.com/PerseusDL/canonical-latinLit/master/data/phi0448/phi001/phi0448.phi001.perseus-lat2.xml

Output: lista di `LatinSegment` con locator gerarchico (book.chapter.section).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests
from lxml import etree

logger = logging.getLogger(__name__)

DBG_LATIN_URL = (
    "https://raw.githubusercontent.com/PerseusDL/canonical-latinLit/master/"
    "data/phi0448/phi001/phi0448.phi001.perseus-lat2.xml"
)

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


@dataclass(frozen=True)
class LatinSegment:
    text: str
    book: int
    chapter: int
    section: int | None  # alcuni paragrafi non sono divisi in sezioni

    @property
    def locator(self) -> str:
        if self.section is None:
            return f"{self.book}.{self.chapter}"
        return f"{self.book}.{self.chapter}.{self.section}"


def fetch_dbg_xml(url: str = DBG_LATIN_URL, timeout: float = 30.0) -> bytes:
    logger.info("Scarico TEI XML del De Bello Gallico da Perseus")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def parse_dbg(xml_bytes: bytes, books: list[int] | None = None) -> list[LatinSegment]:
    """Parsa il TEI XML e restituisce i segmenti latini, opzionalmente
    filtrati per libri (es. `[1, 2]`).

    La struttura TEI di Perseus per il DBG è:
        <div type="edition">
          <div type="textpart" subtype="book" n="1">
            <div type="textpart" subtype="chapter" n="1">
              <div type="textpart" subtype="section" n="1">
                <p>Gallia est omnis...</p>
              </div>
              ...
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
            if section_divs:
                for sec_div in section_divs:
                    sec_n = _safe_int(sec_div.get("n"))
                    text = _collect_text(sec_div)
                    if text:
                        segments.append(LatinSegment(text=text, book=book_n,
                                                     chapter=chap_n, section=sec_n))
            else:
                # Capitolo senza sezioni — prendi il testo intero del capitolo
                text = _collect_text(chap_div)
                if text:
                    segments.append(LatinSegment(text=text, book=book_n,
                                                 chapter=chap_n, section=None))

    logger.info("Parsing completato: %d segmenti latini estratti", len(segments))
    return segments


def _collect_text(elem: etree._Element) -> str:
    """Concatena tutto il testo dei <p> figli, normalizza spazi."""
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
