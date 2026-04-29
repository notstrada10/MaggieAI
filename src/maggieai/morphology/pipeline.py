"""Morphological pipeline for Latin based on CLTK + Stanza.

Exposes a pure `analyze(text)` function that returns a list of tokens
with lemma, POS and morphological features (case, number, tense, ...).

The first invocation downloads the Stanza models for Latin (~500 MB)
into `~/cltk_data`. It is idempotent: subsequent calls are fast.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TokenAnalysis(BaseModel):
    """Morphological analysis of a single token."""

    index: int = Field(description="0-based position of the token in the sentence")
    text: str
    lemma: str | None = None
    pos: str | None = Field(default=None, description="Universal POS tag (NOUN, VERB, ...)")
    features: dict[str, str] = Field(default_factory=dict, description="UD morphological features")
    head: int | None = Field(default=None, description="Head index in the dependency tree")
    dep_rel: str | None = Field(default=None, description="Dependency relation")


class SentenceAnalysis(BaseModel):
    text: str
    tokens: list[TokenAnalysis]


@lru_cache(maxsize=1)
def _get_pipeline() -> Any:  # CLTK NLP type — annotated Any to avoid an import time-bomb
    """Build (lazily) the CLTK pipeline for Latin. Cached per process."""
    from cltk import NLP

    logger.info("Initializing CLTK pipeline for Latin (may require model download)")
    nlp = NLP(language="lat", suppress_banner=True)
    return nlp


def analyze(text: str) -> SentenceAnalysis:
    """Analyze a Latin sentence and return tokens + features.

    Does not handle multi-sentence input: the caller is responsible for
    sentence splitting upstream (or for passing a single sentence).
    """
    nlp = _get_pipeline()
    doc = nlp.analyze(text=text)
    tokens: list[TokenAnalysis] = []
    for i, word in enumerate(doc.words):
        features = _extract_features(word)
        tokens.append(
            TokenAnalysis(
                index=i,
                text=word.string,
                lemma=word.lemma,
                pos=getattr(word, "upos", None) or getattr(word, "pos", None),
                features=features,
                head=getattr(word, "governor", None),
                dep_rel=getattr(word, "dependency_relation", None),
            )
        )
    return SentenceAnalysis(text=text, tokens=tokens)


def _extract_features(word: Any) -> dict[str, str]:
    """Extract morphological features from a CLTK Word object robustly."""
    raw = getattr(word, "features", None)
    if raw is None:
        return {}
    # CLTK exposes features as a MorphosyntacticFeatureBundle (dict-like)
    if hasattr(raw, "items"):
        return {str(k): str(v) for k, v in raw.items() if v is not None}
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items() if v is not None}
    return {}
