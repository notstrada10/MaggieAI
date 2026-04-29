# MaggieAI

> Agentic RAG per il latino classico — non solo traduzione, ma analisi
> filologica con citazioni: lemma-by-lemma morfologia (CLTK), precedenti
> d'autore (Translation Memory su pgvector), regole grammaticali strutturate
> (Allen & Greenough), reasoning loop trasparente.

Stato: **Sprint 1 in corso** — infrastruttura, schema, ingestione dati pilota.
Vedi `~/.claude/plans/blueprint-progetto-lexicon-agentic-wise-piglet.md`
per il blueprint completo.

---

## Architettura in due righe

```
        FastAPI gateway → LangGraph reasoning loop → CLTK morphology + pgvector TM + grammar rules
                              │
                              ├── routing/lightweight  → MLX local (Qwen-2.5-14B, NATIVO macOS)
                              └── translation/critique → Claude Sonnet (API)
```

Quattro decisioni vincolanti del blueprint:

1. **LangGraph** (non LangChain) come orchestratore
2. **RAG-first**, fine-tuning QLoRA solo in v2
3. **MLX inference NATIVA macOS** — Metal non passa via Docker; tutto il
   resto è containerizzato, l'inference no
4. **Embedding multilingue `bge-m3`** (non esistono embedding latini dedicati)

---

## Bootstrap (prima volta)

```bash
# 0. Toolchain
brew install uv

# 1. Dipendenze + venv (uv gestisce Python 3.11 in automatico)
uv sync

# 2. Configurazione
cp .env.example .env
# Modifica .env: imposta ANTHROPIC_API_KEY (resto va bene per dev)

# 3. Stack containerizzato (tutto tranne inference)
docker compose up -d postgres morphology gateway
docker compose logs -f gateway      # verifica che parta pulito

# 4. Schema DB (idempotente — viene anche applicato automaticamente
#    dal primo avvio di Postgres via docker-entrypoint-initdb.d)
docker compose --profile ingest run --rm ingest maggie-ingest init-db

# 5. Carica le 12 regole grammaticali
docker compose --profile ingest run --rm ingest \
    maggie-ingest load-grammar /app/data/grammar_rules

# 6. Inference MLX nativa (terminale separato, sull'host macOS)
uv pip install -e ".[inference-local]"
maggie-inference          # parte mlx-lm.server su :8001 con Qwen-2.5-14B-4bit
                          # primo run scarica ~9GB da HuggingFace
```

## Smoke test

```bash
# 1. Retrieval (Sprint 2 deliverable — non richiede LLM)
curl -s -X POST localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"text": "Gallia est omnis divisa in partes tres", "phenomena": ["genitivo_partitivo"]}' | jq .

# 2. Translate (reasoning loop completo — richiede inference up + ANTHROPIC_API_KEY)
curl -s -X POST localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Caesare imperante, Galli rebellaverunt"}' | jq .
```

Atteso da `/translate`:
- `translation`: traduzione italiana
- `rationale`: motivazione delle scelte morfo-sintattiche
- `morpho_analysis`: token + lemma + POS + features per ciascuna parola
- `phenomena_detected`: es. `["ablativo_assoluto", "participio_congiunto"]`
- `citations`: ≥1 TM hit (se la TM è popolata) + ≥1 regola grammaticale
- `iterations`: 1 o 2 (max — vedi `MAX_ITERATIONS` in `agent/nodes.py`)
- `trace_id`: UUID, indicizzato in `reasoning_traces` per debug

## Layout

```
.
├── pyproject.toml              # uv + tutte le deps + tooling (ruff, mypy, pytest)
├── docker-compose.yml          # postgres + morphology + gateway (no inference)
├── docker/
│   ├── Dockerfile.gateway      # gateway, ingest, eval — stessa immagine
│   └── Dockerfile.morphology   # CLTK + Stanza, volume per i modelli
├── prompts/                    # template Jinja2 per i nodi LLM
├── data/grammar_rules/         # 12 fenomeni in YAML, versionati a mano
├── src/maggieai/
│   ├── config.py               # Pydantic Settings (legge .env)
│   ├── db/                     # schema.sql + SQLAlchemy models + engine
│   ├── inference/              # InferenceClient ABC + MLX local + Claude + router
│   ├── morphology/             # CLTK pipeline + FastAPI service + phenomena detector
│   ├── ingestion/              # Perseus scraper + grammar loader + embedder + CLI
│   ├── agent/                  # LangGraph state, nodes, tools, prompts loader, graph
│   ├── gateway/                # FastAPI app
│   └── eval/                   # gold-set runner + metriche BLEU/chrF
└── tests/                      # unit + integration
```

## Comandi sviluppo

```bash
uv run ruff check src tests          # lint
uv run ruff format src tests         # autoformat
uv run mypy src                      # type check (strict)
uv run pytest                        # test (unit suite, ~0.3s, no DB/LLM)
uv run python -m maggieai.gateway.app  # dev mode (no docker)
```

Suite unit copre i moduli puri (parser TEI, allineatore per locator, matcher
UD-pattern, render Jinja, validatore YAML, helper di parsing JSON e routing
del grafo). Niente DB né LLM: gira ovunque.

## Note operative

- **Inference local non parte? Hai un Mac Intel?** MLX richiede Apple Silicon.
  Soluzione: setta `INFERENCE_ROUTING_MODE=claude-only` in `.env` (in TODO,
  vedi `inference/router.py`).
- **CLTK ci mette 1-2 min al primo `/analyze`**: scarica i modelli Stanza
  per il latino. Sta scaricando in `cltk_data` volume — successivi avvii sono
  istantanei.
- **bge-m3 occupa ~2 GB**: viene scaricato al primo `embed()` nel container
  gateway, persistito in `hf_cache` volume. Idem nel container `ingest`.
- **Limiti dell'aligner v1**: allinea per locator gerarchico
  (`book.chapter.section`). Non funziona se la traduzione non rispetta lo
  stesso schema. Per allineamento sub-sentence usa `vecalign` (TODO v1.5).
- **Regole grammaticali = matcher per-token**: detection è approssimativa
  (es. "ablativo assoluto" fira anche solo su un participio in ablativo).
  Il LLM ha comunque la tabella morfologica completa per validare.

## Sprint roadmap

Vedi blueprint per i dettagli. Riassunto:

| Sprint | Cosa | Stato |
|---|---|---|
| 1 | Infra + ingestione 500 coppie + 12 regole | 🚧 in corso (manca corpus italiano allineato) |
| 2 | RAG vanilla `/retrieve` | ✅ scheletro pronto, da popolare TM |
| 3 | Agent loop `/translate` | ✅ scheletro pronto, da testare end-to-end |
| 4 | Self-critique + eval harness + UI Streamlit | 🟡 parziale (manca UI) |

Coverage attuale unit test: 37 test passanti su 7 moduli (aligner, perseus,
phenomena, prompts, grammar_loader, node helpers, claude `_split_system`).
