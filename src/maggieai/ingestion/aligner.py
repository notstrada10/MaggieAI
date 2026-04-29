"""Allineamento strutturale fra testo latino e traduzione italiana.

**Strategia v1 — semplice ma corretta**: ci affidiamo al fatto che
l'edizione canonica del De Bello Gallico (e tutte le traduzioni serie)
condivide lo stesso schema di numerazione `book.chapter.section`.
Quindi allineiamo per locator: il segmento latino `1.1.1` si aggancia
al segmento italiano `1.1.1`.

**Limitazioni note**:
- Funziona solo per testi canonici con numerazione condivisa.
- Allineamento a livello di sezione, non frase. Per refinement
  sub-sentence (vecalign + LASER) → v1.5.

Per testi senza numerazione condivisa o per allineamento più fine,
sostituire questo modulo con un wrapper su `vecalign`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from maggieai.ingestion.perseus import LatinSegment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ItalianSegment:
    """Segmento italiano caricato a mano o da Wikisource, con stesso locator."""

    text: str
    locator: str  # 'book.chapter[.section]'
    translator: str
    license: str = "PD"  # default: pubblico dominio


@dataclass(frozen=True)
class AlignedPair:
    source_text: str
    target_text: str
    locator: str
    translator: str
    license: str


def align_by_locator(
    latin_segments: list[LatinSegment],
    italian_segments: list[ItalianSegment],
) -> list[AlignedPair]:
    """Allinea per locator esatto. I segmenti senza match vengono droppati con warning."""
    italian_by_loc = {seg.locator: seg for seg in italian_segments}
    pairs: list[AlignedPair] = []
    missing: list[str] = []
    for lat in latin_segments:
        ita = italian_by_loc.get(lat.locator)
        if ita is None:
            missing.append(lat.locator)
            continue
        pairs.append(
            AlignedPair(
                source_text=lat.text,
                target_text=ita.text,
                locator=lat.locator,
                translator=ita.translator,
                license=ita.license,
            )
        )
    if missing:
        logger.warning("Allineamento: %d locator latini senza traduzione (es. %s)",
                       len(missing), missing[:5])
    logger.info("Allineamento: %d coppie create", len(pairs))
    return pairs
