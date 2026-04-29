-- =================================================================
-- MaggieAI — schema iniziale (Sprint 1)
-- =================================================================
-- Idempotente: usa CREATE EXTENSION IF NOT EXISTS / CREATE TABLE IF
-- NOT EXISTS dove possibile. Eseguito automaticamente al primo
-- avvio del container Postgres tramite docker-entrypoint-initdb.d.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- per ricerca BM25-like sui lemmi

-- -----------------------------------------------------------------
-- Tool A: Translation Memory
-- -----------------------------------------------------------------
-- Coppie frase latina ↔ traduzione, con metadata bibliografici.
-- L'embedding è calcolato su `source_text` con bge-m3 (1024-dim).
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
-- Regole sintattiche/morfologiche caricate da YAML. `pattern` è un
-- matcher machine-readable usato dal nodo phenomena_detect; gli altri
-- campi sono prosa per il prompt LLM.
-- -----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS grammar_rules (
    id          BIGSERIAL PRIMARY KEY,
    phenomenon  TEXT          NOT NULL,
    rule_type   TEXT          NOT NULL CHECK (rule_type IN ('syntactic', 'morphological')),
    pattern     JSONB         NOT NULL,
    description TEXT          NOT NULL,
    examples    JSONB         NOT NULL DEFAULT '[]'::jsonb,
    source      TEXT,
    created_at  TIMESTAMPTZ   NOT NULL DEFAULT now(),
    UNIQUE (phenomenon, source)
);

CREATE INDEX IF NOT EXISTS grammar_rules_phenomenon_idx
    ON grammar_rules (phenomenon);

-- -----------------------------------------------------------------
-- Cache morfologica
-- -----------------------------------------------------------------
-- Risultati CLTK per lemmi già visti. Riempita on-demand dal nodo
-- morpho_parse, persiste fra restart.
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
-- Audit log dei reasoning trace
-- -----------------------------------------------------------------
-- Ogni invocazione dell'agente persiste l'intero State. Serve per
-- debugging, eval di regressione e (futuro) curazione dataset
-- per fine-tuning QLoRA.
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
