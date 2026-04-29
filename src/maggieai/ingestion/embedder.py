"""Wrapper attorno a sentence-transformers per BAAI/bge-m3.

Singleton-per-processo: il modello è ~2 GB, lo carichiamo una volta sola.
Per la v1 supportiamo solo l'encoding "dense" — i vettori `colbert` /
`sparse` di bge-m3 sono ignorati per semplicità (potranno essere aggiunti
in v2 se serve hybrid search).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from maggieai.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_model() -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    name = get_settings().embedding_model
    logger.info("Caricamento modello embedding: %s", name)
    return SentenceTransformer(name)


def embed(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Calcola embedding 1024-dim per una lista di stringhe."""
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 64,
        convert_to_numpy=True,
    )
    return vectors.tolist()
