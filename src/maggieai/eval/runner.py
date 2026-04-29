"""Harness di valutazione — Sprint 4 (skeleton).

Per ora implementa solo BLEU/chrF su un gold-set JSONL della forma:
    {"input": "Gallia est ...", "translation": "La Gallia è ...", "phenomena": [...]}
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import click
import httpx
from rich.console import Console
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """Eval CLI per MaggieAI."""


@cli.command("run")
@click.option("--gold", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--gateway-url", default="http://localhost:8000")
@click.option("--limit", type=int, default=None, help="Tronca al primo N record")
def run(gold: Path, gateway_url: str, limit: int | None) -> None:
    """Esegue il gold-set e stampa BLEU + chrF."""
    try:
        from sacrebleu import corpus_bleu, corpus_chrf
    except ImportError as exc:  # pragma: no cover
        raise click.ClickException(
            "sacrebleu non installato. Installa con: uv pip install -e '.[eval]'"
        ) from exc

    records = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        records = records[:limit]

    console.print(f"[bold]Eval su {len(records)} esempi[/bold]")
    hyp: list[str] = []
    ref: list[str] = []

    async def _run_one(client: httpx.AsyncClient, text: str) -> str:
        r = await client.post(f"{gateway_url}/translate", json={"text": text}, timeout=120.0)
        r.raise_for_status()
        return str(r.json()["translation"])

    async def _drive() -> None:
        async with httpx.AsyncClient() as client:
            for rec in records:
                try:
                    out = await _run_one(client, rec["input"])
                except Exception as exc:
                    logger.warning("Errore su input '%s': %s", rec["input"][:50], exc)
                    out = ""
                hyp.append(out)
                ref.append(rec["translation"])
                console.print(f"  [{len(hyp)}/{len(records)}] {rec['input'][:60]}...")

    asyncio.run(_drive())

    bleu = corpus_bleu(hyp, [ref])
    chrf = corpus_chrf(hyp, [ref])

    table = Table(title="Risultati eval")
    table.add_column("Metrica")
    table.add_column("Valore", justify="right")
    table.add_row("BLEU", f"{bleu.score:.2f}")
    table.add_row("chrF", f"{chrf.score:.2f}")
    console.print(table)


if __name__ == "__main__":
    cli()
