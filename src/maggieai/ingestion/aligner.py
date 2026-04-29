"""Structural alignment between Latin text and target translation.

**v1 strategy — simple but correct**: we rely on the fact that the
canonical edition of Caesar's De Bello Gallico (and any serious
translation) shares the same `book.chapter.section` numbering. So we
align by locator: the Latin segment `1.1.1` is paired with the
translated segment `1.1.1`.

**Known limitations**:
- Only works for canonical texts with shared numbering.
- Section-level alignment, not sentence-level. For sub-sentence
  refinement (vecalign + LASER) → v1.5.

For texts without shared numbering or for finer-grained alignment,
replace this module with a `vecalign` wrapper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from maggieai.ingestion.perseus import LatinSegment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranslationSegment:
    """Translated segment loaded by hand or scraped, with the same locator."""

    text: str
    locator: str  # 'book.chapter[.section]'
    translator: str
    license: str = "PD"  # default: public domain


@dataclass(frozen=True)
class AlignedPair:
    source_text: str
    target_text: str
    locator: str
    translator: str
    license: str


def align_by_locator(
    latin_segments: list[LatinSegment],
    translation_segments: list[TranslationSegment],
) -> list[AlignedPair]:
    """Align by exact locator. Latin segments without a match are dropped with a warning."""
    translations_by_loc = {seg.locator: seg for seg in translation_segments}
    pairs: list[AlignedPair] = []
    missing: list[str] = []
    for lat in latin_segments:
        tr = translations_by_loc.get(lat.locator)
        if tr is None:
            missing.append(lat.locator)
            continue
        pairs.append(
            AlignedPair(
                source_text=lat.text,
                target_text=tr.text,
                locator=lat.locator,
                translator=tr.translator,
                license=tr.license,
            )
        )
    if missing:
        logger.warning(
            "Alignment: %d Latin locators without a translation (e.g. %s)",
            len(missing),
            missing[:5],
        )
    logger.info("Alignment: %d pairs created", len(pairs))
    return pairs
