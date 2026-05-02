# Grammar rules — format

Each `*.yaml` file here describes ONE morpho-syntactic phenomenon of
Latin. The loader (`maggie-ingest load-grammar`) imports them
idempotently into the `grammar_rules` table.

## Schema

```yaml
phenomenon: <snake_case slug, unique together with `source`>
rule_type: syntactic | morphological
source: "Allen & Greenough §XYZ"      # optional
translation_template: |               # optional — see below
  "with X V-ing" / "after X V-ed" — pick the connector from context.
description: |
  Explanation in English, detailed enough to be dropped into the LLM
  prompt without further external resources. Light Markdown is fine.
pattern:
  type: ud_pattern
  match_any:                            # OR across the clauses
    - { upos: VERB, Mood: Sub }
    - { lemma_in: [cum, postquam] }
    - { upos: VERB, dep_rel: "acl:relcl" }
examples:
  - lat: "Caesare imperante, Galli rebellaverunt"
    eng: "With Caesar in command, the Gauls rebelled"
    note: "Imperante = present active participle in the ablative"
```

`translation_template` is the canonical English rendering. The loader
splices it into the description as a structured `**Render as:**` block,
which the LLM treats as a stronger signal than the prose description
alone. Keep it short — one or two sentences; mention common confusions
to avoid.

## Supported pattern keys (`ud_pattern`)

The matcher operates per-token. Within a single `match_any` clause all
keys must hold (AND); across clauses any may hold (OR).

- `upos`              — Universal POS tag (NOUN, VERB, ADJ, ADP, SCONJ, …).
- `<UD feature>`      — any UD feature key as it appears in `tok.features`
                        (Case, Mood, Tense, Aspect, Voice, Number, Gender,
                        Person, Degree, VerbForm, …).
- `lemma`             — exact lemma (case-insensitive). Use for closed-class
                        function words (cum, ut, dum, si, …).
- `lemma_in: [...]`   — lemma is one of the listed alternatives. Use for
                        verb-class triggers (verbs of fearing, deponents,
                        impersonals, …).
- `dep_rel`           — Universal Dependencies relation label
                        (`acl:relcl`, `advcl:abs`, `root`, `nsubj`, …).

More sophisticated patterns (sequence, head-of-tree) will be added when
a real case requires them — no preemptive abstractions.
