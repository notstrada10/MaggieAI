# Regole grammaticali — formato

Ogni file `*.yaml` qui dentro descrive UN fenomeno morfo-sintattico
del latino. Il caricatore (`maggie-ingest load-grammar`) li importa
in modo idempotente nella tabella `grammar_rules`.

## Schema

```yaml
phenomenon: <slug snake_case, unique key insieme a `source`>
rule_type: syntactic | morphological
source: "Allen & Greenough §XYZ"      # opzionale
description: |
  Spiegazione in italiano, abbastanza dettagliata da finire dentro
  il prompt LLM senza altre risorse esterne. Markdown leggero ok.
pattern:
  type: ud_pattern
  match_any:                            # OR fra le clausole
    - { upos: VERB, Mood: Sub }
    - { upos: NOUN, Case: Abl }
examples:
  - lat: "Caesare imperante, Galli rebellaverunt"
    ita: "Comandando Cesare, i Galli si ribellarono"
    note: "Imperante = participio presente in ablativo"
```

## `pattern.type` supportati (v1)

- `ud_pattern`: matcha per-token su `upos` (Universal POS) e features UD.
  Il match è OR fra le clausole di `match_any`. Per-clausola è AND
  fra le condizioni del dict.

Pattern più sofisticati (sequence, dep tree) verranno aggiunti
quando un caso reale lo richiederà — niente astrazioni preventive.
