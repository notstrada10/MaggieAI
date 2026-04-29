"""Caricamento template Jinja2 dai file in `prompts/`.

Tenere i prompt fuori dal codice Python permette di:
- versionarli/diff-arli leggibilmente
- iterare sui prompt senza riavviare i servizi
- caricare versioni multiple per A/B test futuri
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

# `prompts/` è alla root del repo, non dentro il package
_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(_PROMPTS_DIR),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def render(template_name: str, **context: Any) -> str:
    return _env().get_template(template_name).render(**context)
