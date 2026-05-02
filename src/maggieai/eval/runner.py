"""Evaluation harness — Sprint 4 (skeleton).

For now it only computes BLEU/chrF on a JSONL gold set of the form:
    {"input": "Gallia est ...", "translation": "All Gaul is ...", "phenomena": [...]}
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

import click
import httpx
from rich.console import Console
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)

RESPONDEO_TRANSLATION_URL = (
    "https://raw.githubusercontent.com/slanglab/RespondeoQA/main/"
    "data/final_dataset/junior_scholarship_translation_long.json"
)


@click.group()
def cli() -> None:
    """MaggieAI eval CLI."""


def _extract_latin(question_text: str) -> str | None:
    """Pull the Latin out of a RespondeoQA translation question.

    Two patterns observed:
      - Italicized:   'Translate—*<latin>.*'  (most common)
      - Bare:         'Translate: <latin>.'   (no italics)
    """
    italics = re.search(r"\*([^*]+?)\*", question_text)
    if italics:
        return italics.group(1).strip() or None
    stripped = re.sub(
        r"^(Derive and translate|Translate)[\s—–\-:]+",
        "",
        question_text.strip(),
        flags=re.IGNORECASE,
    )
    return stripped.strip() or None


@cli.command("prepare-respondeo")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("eval/respondeo_lat_en.jsonl"),
    show_default=True,
    help="Where to write the gold JSONL.",
)
def prepare_respondeo(output: Path) -> None:
    """Build a Latin→English gold set from RespondeoQA's junior_scholarship_translation_long.

    Filters to translation items that go Latin→English (where the question wraps
    the Latin in italics or a 'Translate—' prefix and the answer is the English
    rendering). Writes one JSON object per line: input, translation, phenomena, question_id.
    """
    console.print(f"[bold]Downloading[/bold] {RESPONDEO_TRANSLATION_URL}")
    resp = httpx.get(RESPONDEO_TRANSLATION_URL, timeout=30.0)
    resp.raise_for_status()
    items = resp.json()

    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_direction = 0
    skipped_extract = 0

    with output.open("w", encoding="utf-8") as fh:
        for item in items:
            if (
                item.get("answer_language") != "english"
                or item.get("question_content") != "translation"
            ):
                skipped_direction += 1
                continue
            latin = _extract_latin(item.get("question_text", ""))
            answers = item.get("answers") or []
            if not latin or not answers:
                skipped_extract += 1
                continue
            record = {
                "input": latin,
                "translation": answers[0],
                "phenomena": [],
                "question_id": item.get("question_id"),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    console.print(f"[green]Wrote[/green] {written} records → {output}")
    console.print(
        f"[dim]Skipped {skipped_direction} (wrong direction), "
        f"{skipped_extract} (extraction failed)[/dim]"
    )


@cli.command("run")
@click.option("--gold", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--gateway-url", default="http://localhost:18000")
@click.option("--limit", type=int, default=None, help="Truncate to the first N records")
def run(gold: Path, gateway_url: str, limit: int | None) -> None:
    """Run the gold set and print BLEU + chrF."""
    try:
        from sacrebleu import corpus_bleu, corpus_chrf
    except ImportError as exc:  # pragma: no cover
        raise click.ClickException(
            "sacrebleu is not installed. Install with: uv pip install -e '.[eval]'"
        ) from exc

    records = [
        json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if limit:
        records = records[:limit]

    console.print(f"[bold]Eval over {len(records)} examples[/bold]")
    hyp: list[str] = []
    ref: list[str] = []

    async def _run_one(client: httpx.AsyncClient, text: str) -> str:
        r = await client.post(f"{gateway_url}/translate", json={"text": text}, timeout=600.0)
        r.raise_for_status()
        return str(r.json()["translation"])

    async def _drive() -> None:
        async with httpx.AsyncClient() as client:
            for rec in records:
                try:
                    out = await _run_one(client, rec["input"])
                except Exception as exc:
                    logger.warning("Error on input '%s': %s", rec["input"][:50], exc)
                    out = ""
                hyp.append(out)
                ref.append(rec["translation"])
                console.print(f"  [{len(hyp)}/{len(records)}] {rec['input'][:60]}...")

    asyncio.run(_drive())

    bleu = corpus_bleu(hyp, [ref])
    chrf = corpus_chrf(hyp, [ref])

    table = Table(title="Eval results")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("BLEU", f"{bleu.score:.2f}")
    table.add_row("chrF", f"{chrf.score:.2f}")
    console.print(table)


if __name__ == "__main__":
    cli()
