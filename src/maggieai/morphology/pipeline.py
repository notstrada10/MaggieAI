"""Pipeline morfologica per il latino basata su CLTK + Stanza.

Espone una funzione pura `analyze(text)` che restituisce una lista di
token con lemma, POS e features morfologiche (caso, numero, tempo, ...).

Il primo invocazione scarica i modelli Stanza per il latino (~500 MB)
in `~/cltk_data`. È idempotente: chiamate successive sono veloci.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TokenAnalysis(BaseModel):
    """Analisi morfologica di un singolo token."""

    index: int = Field(description="Posizione 0-based del token nella frase")
    text: str
    lemma: str | None = None
    pos: str | None = Field(default=None, description="Universal POS tag (NOUN, VERB, ...)")
    features: dict[str, str] = Field(default_factory=dict, description="Features morfologiche UD")
    head: int | None = Field(default=None, description="Indice del head nella dependency tree")
    dep_rel: str | None = Field(default=None, description="Relazione di dipendenza")


class SentenceAnalysis(BaseModel):
    text: str
    tokens: list[TokenAnalysis]


@lru_cache(maxsize=1)
def _get_pipeline() -> Any:  # CLTK NLP type — annotato Any per evitare import time-bomb
    """Crea (lazy) la pipeline CLTK per il latino. Cached per processo."""
    from cltk import NLP

    logger.info("Inizializzazione pipeline CLTK per il latino (può richiedere il download dei modelli)")
    nlp = NLP(language="lat", suppress_banner=True)
    return nlp


def analyze(text: str) -> SentenceAnalysis:
    """Analizza una frase latina e restituisce token+features.

    Non gestisce input multi-frase: il chiamante deve fare il sentence
    splitting a monte (oppure passare una frase intera).
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
    """Estrae le features morfologiche dall'oggetto Word di CLTK in modo robusto."""
    raw = getattr(word, "features", None)
    if raw is None:
        return {}
    # CLTK espone le features come MorphosyntacticFeatureBundle (dict-like)
    if hasattr(raw, "items"):
        return {str(k): str(v) for k, v in raw.items() if v is not None}
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items() if v is not None}
    return {}
