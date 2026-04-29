"""Wrapper around sentence-transformers for BAAI/bge-m3.

Per-process singleton: the model is ~2 GB, we load it only once.
For v1 we only support the "dense" encoding — bge-m3's `colbert` /
`sparse` vectors are ignored for simplicity (they could be added in
v2 if hybrid search is needed).
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
    logger.info("Loading embedding model: %s", name)
    return SentenceTransformer(name)


def embed(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Compute 1024-dim embeddings for a list of strings."""
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 64,
        convert_to_numpy=True,
    )
    return vectors.tolist()
