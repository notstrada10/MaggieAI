"""SQLAlchemy models for the tables defined in `schema.sql`.

The authoritative schema is the SQL file — these models are a wrapper
for type-safe queries in Python code. If you update the schema, keep
this file in sync.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from maggieai.config import get_settings

_EMBEDDING_DIM = get_settings().embedding_dim


class Base(DeclarativeBase):
    pass


class TranslationPair(Base):
    __tablename__ = "translation_pairs"
    __table_args__ = (
        UniqueConstraint(
            "author", "work", "locator", name="translation_pairs_author_work_locator_uq"
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    target_text: Mapped[str] = mapped_column(Text, nullable=False)
    author: Mapped[str | None] = mapped_column(Text)
    work: Mapped[str | None] = mapped_column(Text)
    locator: Mapped[str | None] = mapped_column(Text)
    translator: Mapped[str | None] = mapped_column(Text)
    license: Mapped[str | None] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(_EMBEDDING_DIM))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class GrammarRule(Base):
    __tablename__ = "grammar_rules"
    __table_args__ = (
        CheckConstraint(
            "rule_type IN ('syntactic', 'morphological')", name="grammar_rule_type_chk"
        ),
        UniqueConstraint("phenomenon", "source", name="grammar_rules_phenom_source_uq"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    phenomenon: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    rule_type: Mapped[str] = mapped_column(String(32), nullable=False)
    pattern: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    examples: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    source: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class LemmaCache(Base):
    __tablename__ = "lemma_cache"

    lemma: Mapped[str] = mapped_column(Text, primary_key=True)
    pos: Mapped[str | None] = mapped_column(Text)
    features: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    glosses_it: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    source: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ReasoningTrace(Base):
    __tablename__ = "reasoning_traces"

    trace_id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    state_dump: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    user_rating: Mapped[int | None] = mapped_column(SmallInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
