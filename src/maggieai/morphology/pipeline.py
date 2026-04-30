"""Morphological pipeline for Latin based on CLTK + Stanza.

Exposes a pure `analyze(text)` function that returns a list of tokens
with lemma, POS and morphological features (case, number, tense, ...).

The first invocation downloads the Stanza models for Latin (~500 MB)
into `~/cltk_data`. It is idempotent: subsequent calls are fast.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TokenAnalysis(BaseModel):
    """Morphological analysis of a single token."""

    index: int = Field(description="0-based position of the token in the sentence")
    text: str
    lemma: str | None = None
    pos: str | None = Field(default=None, description="Universal POS tag (NOUN, VERB, ...)")
    features: dict[str, str] = Field(default_factory=dict, description="UD morphological features")
    head: int | None = Field(default=None, description="Head index in the dependency tree")
    dep_rel: str | None = Field(default=None, description="Dependency relation")


class SentenceAnalysis(BaseModel):
    text: str
    tokens: list[TokenAnalysis]


@lru_cache(maxsize=1)
def _get_pipeline() -> Any:  # CLTK NLP type — annotated Any to avoid an import time-bomb
    """Build (lazily) the CLTK pipeline for Latin. Cached per process.

    Three CLTK warts handled here:

    1. Several CLTK processes (Stanza wrapper, FastText embeddings, ...)
       prompt the user via ``input()`` to confirm a model download. That
       crashes with ``EOFError`` when running as a FastAPI service (no
       stdin). We monkey-patch ``cltk.utils.utils.query_yes_no`` once to
       auto-yes — covers every prompt CLTK might raise downstream.
    2. CLTK's non-interactive path calls ``stanza.download()`` without
       passing ``model_dir``, so the model lands in stanza's default
       cache (~/Library/Caches/stanza on macOS), but CLTK checks
       ``~/stanza_resources/...`` and raises FileNotFoundError. We
       pre-download Stanza directly to the path CLTK expects.
    3. CLTK's Latin pipeline pulls FastText embeddings by default. We
       don't use the word vectors anywhere, but the download still fires
       on first ``analyze()``. Auto-yes handles it; future optimization
       could prune the embeddings process from the pipeline.
    """
    import os
    from pathlib import Path

    import stanza
    from cltk import NLP

    # 1. Make every CLTK download prompt auto-yes. CLTK's `query_yes_no`
    # is imported by 7+ modules (and grows as CLTK adds features), so
    # patching each binding is fragile. Instead we patch `builtins.input`,
    # which `query_yes_no` calls internally — works regardless of which
    # module raised the prompt or when it was imported.
    import builtins

    builtins.input = lambda *_args, **_kwargs: "yes"  # type: ignore[assignment]

    # 2. Pre-download Stanza to the directory CLTK actually checks
    cltk_models_dir = Path("~/stanza_resources").expanduser()
    expected_pt = cltk_models_dir / "la" / "tokenize" / "ittb.pt"
    if not expected_pt.exists():
        logger.info("Downloading Stanza Latin (ittb) model to %s", cltk_models_dir)
        os.makedirs(cltk_models_dir, exist_ok=True)
        stanza.download(lang="la", package="ittb", model_dir=str(cltk_models_dir))

    logger.info("Initializing CLTK pipeline for Latin")
    nlp = NLP(language="lat", suppress_banner=True)
    return nlp


def analyze(text: str) -> SentenceAnalysis:
    """Analyze a Latin sentence and return tokens + features.

    Does not handle multi-sentence input: the caller is responsible for
    sentence splitting upstream (or for passing a single sentence).
    """
    nlp = _get_pipeline()
    doc = nlp.analyze(text=text)
    tokens: list[TokenAnalysis] = []
    for i, word in enumerate(doc.words):
        features = _extract_features(word)
        tokens.append(
            TokenAnalysis(
                index=i,
                text=word.string,
                lemma=word.lemma,
                pos=getattr(word, "upos", None) or getattr(word, "pos", None),
                features=features,
                head=getattr(word, "governor", None),
                dep_rel=getattr(word, "dependency_relation", None),
            )
        )
    return SentenceAnalysis(text=text, tokens=tokens)


# Map CLTK's bracketed-lowercase feature values to UD spec abbreviations.
# CLTK exposes Latin morphology as a MorphosyntacticFeatureBundle whose
# values stringify to forms like ``[ablative]`` and ``[participle]``. The
# YAML grammar-rule patterns in ``data/grammar_rules`` follow the
# Universal Dependencies spec (``Abl``, ``Part``). We translate at this
# boundary so the rest of the pipeline (phenomena detector, prompt
# templates, LLM evidence) sees a single, documented format.
_UD_VALUE_MAP: dict[str, str] = {
    # Case
    "ablative": "Abl",
    "accusative": "Acc",
    "dative": "Dat",
    "genitive": "Gen",
    "locative": "Loc",
    "nominative": "Nom",
    "vocative": "Voc",
    # VerbForm
    "participle": "Part",
    "infinitive": "Inf",
    "finite": "Fin",
    "gerundive": "Gdv",
    "gerund": "Ger",
    "supine": "Sup",
    # Mood
    "indicative": "Ind",
    "subjunctive": "Sub",
    "imperative": "Imp",
    # Voice
    "active": "Act",
    "passive": "Pass",
    # Number
    "singular": "Sing",
    "plural": "Plur",
    "dual": "Dual",
    # Gender
    "masculine": "Masc",
    "feminine": "Fem",
    "neuter": "Neut",
    # Person
    "first": "1",
    "second": "2",
    "third": "3",
    # Aspect (Latin: CLTK uses Aspect=imperfective for the imperfect tense)
    "imperfective": "Imp",
    "perfective": "Perf",
    "progressive": "Prog",
    # Tense
    "present": "Pres",
    "past": "Past",
    "future": "Fut",
    "imperfect": "Imp",
    "pluperfect": "Pqp",
}

# Only normalize values for keys we actually match against in the patterns.
# Other CLTK keys (InflClass, ...) pass through verbatim — useful context
# for the LLM, ignored by the matcher.
_NORMALIZABLE_KEYS: frozenset[str] = frozenset(
    {"Case", "VerbForm", "Mood", "Voice", "Number", "Gender", "Person", "Aspect", "Tense"}
)


def _normalize_ud_value(raw: str) -> str:
    """Strip CLTK brackets and map a single value to its UD abbreviation.

    Returns the input unchanged when:
    - the value is multi-valued (e.g. ``[lat_a, nominal]``), since these
      are CLTK-specific bundles that don't correspond to a UD value;
    - the lowercase token is not in the map (forward compatibility with
      future CLTK additions or rare features).
    """
    s = raw.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if "," in s:
        return raw
    return _UD_VALUE_MAP.get(s.lower(), s)


def _extract_features(word: Any) -> dict[str, str]:
    """Extract and UD-normalize morphological features from a CLTK Word.

    Pattern matchers in :mod:`maggieai.morphology.phenomena` compare
    feature values via string equality, so keeping CLTK and YAML in the
    same spec is necessary — see :data:`_UD_VALUE_MAP`.
    """
    raw = getattr(word, "features", None)
    if raw is None:
        return {}
    if hasattr(raw, "items"):
        items = raw.items()
    elif isinstance(raw, dict):
        items = raw.items()
    else:
        return {}
    out: dict[str, str] = {}
    for k, v in items:
        if v is None:
            continue
        key = str(k)
        value = str(v)
        if key in _NORMALIZABLE_KEYS:
            value = _normalize_ud_value(value)
        out[key] = value
    return out
