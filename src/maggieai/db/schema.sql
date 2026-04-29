-- =================================================================
-- MaggieAI — initial schema (Sprint 1)
-- =================================================================
-- Idempotent: uses CREATE EXTENSION IF NOT EXISTS / CREATE TABLE IF
-- NOT EXISTS where possible. Applied automatically on the first start
-- of the Postgres container via docker-entrypoint-initdb.d.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- for BM25-like search on lemmas

-- -----------------------------------------------------------------
-- Tool A: Translation Memory
-- -----------------------------------------------------------------
-- Latin sentence ↔ translation pairs, with bibliographic metadata.
-- The embedding is computed on `source_text` with bge-m3 (1024-dim).
-- -----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS translation_pairs (
    id          BIGSERIAL PRIMARY KEY,
    source_text TEXT          NOT NULL,
    target_text TEXT          NOT NULL,
    author      TEXT,
    work        TEXT,
    locator     TEXT,
    translator  TEXT,
    license     TEXT,
    embedding   VECTOR(1024),
    created_at  TIMESTAMPTZ   NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS translation_pairs_embedding_hnsw_idx
    ON translation_pairs USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS translation_pairs_author_work_idx
    ON translation_pairs (author, work);

CREATE INDEX IF NOT EXISTS translation_pairs_source_trgm_idx
    ON translation_pairs USING gin (source_text gin_trgm_ops);

-- -----------------------------------------------------------------
-- Tool B: Grammar Engine
-- -----------------------------------------------------------------
-- Syntactic/morphological rules loaded from YAML. `pattern` is a
-- machine-readable matcher used by the phenomena_detect node; the
-- other fields are prose for the LLM prompt.
-- -----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS grammar_rules (
    id          BIGSERIAL PRIMARY KEY,
    phenomenon  TEXT          NOT NULL,
    rule_type   TEXT          NOT NULL,
    pattern     JSONB         NOT NULL,
    description TEXT          NOT NULL,
    examples    JSONB         NOT NULL DEFAULT '[]'::jsonb,
    source      TEXT,
    created_at  TIMESTAMPTZ   NOT NULL DEFAULT now(),
    -- Constraints must be named explicitly so SQLAlchemy ON CONFLICT clauses
    -- can reference them by name (see grammar_loader.load_directory).
    CONSTRAINT grammar_rule_type_chk
        CHECK (rule_type IN ('syntactic', 'morphological')),
    CONSTRAINT grammar_rules_phenom_source_uq
        UNIQUE (phenomenon, source)
);

CREATE INDEX IF NOT EXISTS grammar_rules_phenomenon_idx
    ON grammar_rules (phenomenon);

-- -----------------------------------------------------------------
-- Morphology cache
-- -----------------------------------------------------------------
-- CLTK results for already-seen lemmas. Filled on demand by the
-- morpho_parse node, persisted across restarts.
-- -----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS lemma_cache (
    lemma      TEXT PRIMARY KEY,
    pos        TEXT,
    features   JSONB,
    glosses_it TEXT[],
    source     TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- -----------------------------------------------------------------
-- Reasoning trace audit log
-- -----------------------------------------------------------------
-- Each agent invocation persists the entire State. Used for
-- debugging, regression eval, and (future) dataset curation for
-- QLoRA fine-tuning.
-- -----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reasoning_traces (
    trace_id    UUID PRIMARY KEY,
    input_text  TEXT          NOT NULL,
    state_dump  JSONB         NOT NULL,
    user_rating SMALLINT      CHECK (user_rating BETWEEN 1 AND 5),
    created_at  TIMESTAMPTZ   NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS reasoning_traces_created_at_idx
    ON reasoning_traces (created_at DESC);
