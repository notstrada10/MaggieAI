"""Tool del reasoning loop: Translation Memory + Grammar Engine.

Sono wrapper sottili sul DB Postgres. Usati dentro i nodi del grafo
(non come `langchain.Tool` di prima generazione — non ci serve quel
livello di astrazione perché siamo già dentro LangGraph).
"""

from __future__ import annotations

import logging

from sqlalchemy import select, text

from maggieai.agent.state import GrammarHit, TMHit
from maggieai.db.engine import session_scope
from maggieai.db.models import GrammarRule, TranslationPair

logger = logging.getLogger(__name__)


def tm_lookup(query_embedding: list[float], k: int = 5) -> list[TMHit]:
    """Top-k semantica su `translation_pairs` via pgvector (cosine)."""
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
    """Recupera le regole grammaticali per i `phenomenon` indicati."""
    if not phenomena:
        return []
    with session_scope() as session:
        rows = session.execute(
            select(GrammarRule).where(GrammarRule.phenomenon.in_(phenomena))
        ).scalars().all()
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
    """Restituisce tutti i pattern usati dal nodo `phenomena_detect`."""
    with session_scope() as session:
        rows = session.execute(
            select(GrammarRule.phenomenon, GrammarRule.pattern)
        ).all()
    return [{"phenomenon": p, "pattern": pat} for p, pat in rows]
