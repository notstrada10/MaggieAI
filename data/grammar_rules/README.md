# Grammar rules — format

Each `*.yaml` file here describes ONE morpho-syntactic phenomenon of
Latin. The loader (`maggie-ingest load-grammar`) imports them
idempotently into the `grammar_rules` table.

## Schema

```yaml
phenomenon: <snake_case slug, unique together with `source`>
rule_type: syntactic | morphological
source: "Allen & Greenough §XYZ"      # optional
description: |
  Explanation in English, detailed enough to be dropped into the LLM
  prompt without further external resources. Light Markdown is fine.
pattern:
  type: ud_pattern
  match_any:                            # OR across the clauses
    - { upos: VERB, Mood: Sub }
    - { upos: NOUN, Case: Abl }
examples:
  - lat: "Caesare imperante, Galli rebellaverunt"
    eng: "With Caesar in command, the Gauls rebelled"
    note: "Imperante = present active participle in the ablative"
```

## Supported `pattern.type` (v1)

- `ud_pattern`: matches per-token on `upos` (Universal POS) and UD
  features. Match is OR across the `match_any` clauses. Within a
  single clause it is AND across the dict conditions.

More sophisticated patterns (sequence, dependency tree) will be added
when a real case requires them — no preemptive abstractions.
