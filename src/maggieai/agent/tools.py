"""Reasoning-loop tools: Translation Memory + Grammar Engine.

These are thin wrappers over Postgres. Used inside the graph nodes
(not as first-generation `langchain.Tool` — we don't need that level
of abstraction because we are already inside LangGraph).
"""

from __future__ import annotations

import logging

from sqlalchemy import select, text

from maggieai.agent.state import GrammarHit, TMHit
from maggieai.db.engine import session_scope
from maggieai.db.models import GrammarRule

logger = logging.getLogger(__name__)


def tm_lookup(query_embedding: list[float], k: int = 5) -> list[TMHit]:
    """Top-k semantic search on `translation_pairs` via pgvector (cosine)."""
    sql = text(
        """
        SELECT source_text, target_text, author, work, locator, translator,
               (embedding <=> CAST(:vec AS vector)) AS distance
        FROM translation_pairs
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> CAST(:vec AS vector)
        LIMIT :k
        """
    )
    with session_scope() as session:
        rows = session.execute(sql, {"vec": query_embedding, "k": k}).mappings().all()
    return [
        TMHit(
            source_text=row["source_text"],
            target_text=row["target_text"],
            author=row["author"],
            work=row["work"],
            locator=row["locator"],
            translator=row["translator"],
            distance=float(row["distance"]),
        )
        for row in rows
    ]


def grammar_lookup(phenomena: list[str]) -> list[GrammarHit]:
    """Fetch the grammar rules for the given `phenomenon` slugs."""
    if not phenomena:
        return []
    with session_scope() as session:
        rows = (
            session.execute(select(GrammarRule).where(GrammarRule.phenomenon.in_(phenomena)))
            .scalars()
            .all()
        )
    return [
        GrammarHit(
            phenomenon=row.phenomenon,
            description=row.description,
            examples=row.examples,
            source=row.source,
        )
        for row in rows
    ]


def all_grammar_patterns() -> list[dict[str, object]]:
    """Return all the patterns used by the `phenomena_detect` node."""
    with session_scope() as session:
        rows = session.execute(select(GrammarRule.phenomenon, GrammarRule.pattern)).all()
    return [{"phenomenon": p, "pattern": pat} for p, pat in rows]
