# MaggieAI

> Agentic RAG for Classical Latin — not just translation, but
> philological analysis with citations: lemma-by-lemma morphology
> (CLTK), authorial precedents (Translation Memory on pgvector),
> structured grammar rules (Allen & Greenough), transparent reasoning
> loop.

Status: **Sprint 1 in progress** — infrastructure, schema, pilot data
ingestion. See `~/.claude/plans/blueprint-progetto-lexicon-agentic-wise-piglet.md`
for the full blueprint.

---

## Architecture in two lines

```
        FastAPI gateway → LangGraph reasoning loop → CLTK morphology + pgvector TM + grammar rules
                              │
                              ├── routing/lightweight  → MLX local (Qwen-2.5-14B, NATIVE macOS)
                              └── translation/critique → Claude Sonnet (API)
```

Four binding decisions from the blueprint:

1. **LangGraph** (not LangChain) as orchestrator
2. **RAG-first**, QLoRA fine-tuning only in v2
3. **MLX inference NATIVE on macOS** — Metal does not pass through
   Docker; everything else is containerized, inference is not
4. **Multilingual embedding `bge-m3`** (no dedicated Latin embeddings exist)

Output language: **English** (decision taken — see open question B in
the blueprint, now resolved). Prompts, grammar rule descriptions and
the future TM corpus all target English.

---

## Bootstrap (first time)

```bash
# 0. Toolchain
brew install uv

# 1. Dependencies + venv (uv manages Python 3.11 automatically)
uv sync

# 2. Configuration
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY (the rest is fine for dev)

# 3. Containerized stack (everything except inference)
docker compose up -d postgres morphology gateway
docker compose logs -f gateway      # check that it starts cleanly

# 4. DB schema (idempotent — also applied automatically on the
#    first start of Postgres via docker-entrypoint-initdb.d)
docker compose --profile ingest run --rm ingest maggie-ingest init-db

# 5. Load the 12 grammar rules
docker compose --profile ingest run --rm ingest \
    maggie-ingest load-grammar /app/data/grammar_rules

# 6. Native MLX inference (separate terminal, on the macOS host)
uv pip install -e ".[inference-local]"
maggie-inference          # starts mlx-lm.server on :8001 with Qwen-2.5-14B-4bit
                          # first run downloads ~9GB from HuggingFace
```

## Smoke test

```bash
# 1. Retrieval (Sprint 2 deliverable — does not require an LLM)
curl -s -X POST localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"text": "Gallia est omnis divisa in partes tres", "phenomena": ["genitivo_partitivo"]}' | jq .

# 2. Translate (full reasoning loop — requires inference up + ANTHROPIC_API_KEY)
curl -s -X POST localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Caesare imperante, Galli rebellaverunt"}' | jq .
```

Expected from `/translate`:
- `translation`: English translation
- `rationale`: justification of morpho-syntactic choices
- `morpho_analysis`: token + lemma + POS + features for each word
- `phenomena_detected`: e.g. `["ablativo_assoluto", "participio_congiunto"]`
- `citations`: ≥1 TM hit (if the TM is populated) + ≥1 grammar rule
- `iterations`: 1 or 2 (max — see `MAX_ITERATIONS` in `agent/nodes.py`)
- `trace_id`: UUID, indexed in `reasoning_traces` for debugging

## Layout

```
.
├── pyproject.toml              # uv + all deps + tooling (ruff, mypy, pytest)
├── docker-compose.yml          # postgres + morphology + gateway (no inference)
├── docker/
│   ├── Dockerfile.gateway      # gateway, ingest, eval — same image
│   └── Dockerfile.morphology   # CLTK + Stanza, volume for the models
├── prompts/                    # Jinja2 templates for the LLM nodes
├── data/grammar_rules/         # 12 phenomena in YAML, hand-versioned
├── src/maggieai/
│   ├── config.py               # Pydantic Settings (reads .env)
│   ├── db/                     # schema.sql + SQLAlchemy models + engine
│   ├── inference/              # InferenceClient ABC + MLX local + Claude + router
│   ├── morphology/             # CLTK pipeline + FastAPI service + phenomena detector
│   ├── ingestion/              # Perseus scraper + grammar loader + embedder + CLI
│   ├── agent/                  # LangGraph state, nodes, tools, prompts loader, graph
│   ├── gateway/                # FastAPI app
│   └── eval/                   # gold-set runner + BLEU/chrF metrics
└── tests/                      # unit + integration
```

## Dev commands

```bash
uv run ruff check src tests          # lint
uv run ruff format src tests         # autoformat
uv run mypy src                      # type check (strict)
uv run pytest                        # tests (unit suite, ~0.3s, no DB/LLM)
uv run python -m maggieai.gateway.app  # dev mode (no docker)
```

The unit suite covers the pure modules (TEI parser, locator aligner,
UD-pattern matcher, Jinja render, YAML validator, JSON-parsing helpers
and graph routing). No DB, no LLM: runs anywhere.

## Operational notes

- **Local inference does not start? You have an Intel Mac?** MLX
  requires Apple Silicon. Set `INFERENCE_ROUTING_MODE=claude-only` (or
  `deepseek-only`) in `.env`, or pick the mode per-request in the
  Workbench sidebar.
- **CLTK takes 1–2 min on the first `/analyze`**: it downloads the
  Stanza models for Latin into the `cltk_data` volume — subsequent
  starts are instant.
- **bge-m3 is ~2 GB**: downloaded on the first `embed()` inside the
  gateway container, persisted in the `hf_cache` volume. Same in the
  `ingest` container.
- **Aligner v1 limits**: aligns by hierarchical locator
  (`book.chapter.section`). Does not work if the translation does not
  follow the same scheme. For sub-sentence alignment use `vecalign`
  (TODO v1.5).
- **Grammar rules = per-token matchers**: detection is approximate
  (e.g. "ablative absolute" fires on any participle in the ablative).
  The LLM still has the full morphology table to validate.

## Sprint roadmap

See blueprint for details. Summary:

| Sprint | What | Status |
|---|---|---|
| 1 | Infra + 500-pair ingestion + 12 rules | ✅ done (647 Caesar pairs loaded — DBG 404 + DBC 243; 12 rules in DB) |
| 2 | RAG vanilla `/retrieve` | ✅ done (pgvector HNSW + trigram source index live) |
| 3 | Agent loop `/translate` | ✅ done (end-to-end traces in `reasoning_traces`) |
| 4 | Self-critique + eval harness + Streamlit UI | 🟡 partial — UI shipped, eval harness has Qwen-14B-local full numbers (BLEU 7.76 / chrF 28.96 on RespondeoQA 174); full Claude run still pending |

Current unit-test coverage: 69 tests passing.

## Workbench (Streamlit)

```bash
# Gateway must be reachable (default: localhost:18000)
GATEWAY_URL=http://localhost:18000 \
  uv run streamlit run src/maggieai/ui/app.py
```

Features:
- **Routing selector** in the sidebar — switch between Hybrid / Claude
  only / Local only / DeepSeek only without touching `.env`. The choice
  is sent as `routing_mode` on each `POST /translate`.
- **Token-click → morphology highlight**: click any word in the
  rendered source line to highlight its row in the Morphology tab.
- **TM citations show the matched Latin and English snippets**
  alongside the locator and cosine distance.
- **Rating** (I–V) under each translation; written to
  `reasoning_traces.user_rating` via `PATCH /traces/{id}/rating`. Ratings
  are visible next to each entry in the Recent traces list.
- **Compare mode** — side-by-side two historical traces for spotting
  regressions when changing prompts or models.
