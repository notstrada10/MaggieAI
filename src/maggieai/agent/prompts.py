"""Loading of Jinja2 templates from files in `prompts/`.

Keeping prompts outside Python lets us:
- version/diff them readably
- iterate on prompts without restarting services
- load multiple versions for future A/B testing
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

# `prompts/` lives at the repo root, not inside the package
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
