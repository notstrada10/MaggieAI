"""CLI di ingestione (alias `maggie-ingest`).

Comandi:
    maggie-ingest init-db                     # applica schema.sql
    maggie-ingest load-grammar [PATH]         # carica regole YAML
    maggie-ingest load-corpus dbg --books 1,2 # scarica e ingesta DBG
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Logging DEBUG")
def cli(verbose: bool) -> None:
    """Ingestione dati per MaggieAI."""
    _setup_logging(verbose)


@cli.command("init-db")
def init_db() -> None:
    """Applica `schema.sql` al database configurato."""
    from sqlalchemy import text

    from maggieai.db.engine import get_engine

    schema_path = Path(__file__).parents[1] / "db" / "schema.sql"
    sql = schema_path.read_text(encoding="utf-8")
    with get_engine().begin() as conn:
        # psycopg accetta script multi-statement quando passati così
        conn.exec_driver_sql(sql)
    console.print(f"[green]✓[/green] Schema applicato da {schema_path}")


@cli.command("load-grammar")
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/grammar_rules"),
)
def load_grammar(path: Path) -> None:
    """Carica le regole YAML in `grammar_rules`."""
    from maggieai.ingestion.grammar_loader import load_directory

    n = load_directory(path)
    console.print(f"[green]✓[/green] Caricate {n} regole grammaticali")


@cli.command("load-corpus")
@click.argument("corpus", type=click.Choice(["dbg"]))
@click.option(
    "--books",
    default="1,2",
    help="CSV di numeri di libro (es. '1,2,3')",
)
@click.option(
    "--italian-jsonl",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="JSONL con la traduzione italiana (locator+text+translator). "
    "Se omesso, ingesta solo testo latino senza coppie allineate.",
)
def load_corpus(corpus: str, books: str, italian_jsonl: Path | None) -> None:
    """Scarica un corpus, allinea con la traduzione e popola la TM."""
    import json

    from sqlalchemy import insert

    from maggieai.db.engine import session_scope
    from maggieai.db.models import TranslationPair
    from maggieai.ingestion.aligner import ItalianSegment, align_by_locator
    from maggieai.ingestion.embedder import embed
    from maggieai.ingestion.perseus import fetch_dbg_xml, parse_dbg

    if corpus != "dbg":
        raise click.ClickException(f"Corpus sconosciuto: {corpus}")

    book_list = [int(b.strip()) for b in books.split(",") if b.strip()]
    console.print(f"[bold]Ingestione DBG, libri {book_list}[/bold]")

    xml = fetch_dbg_xml()
    latin_segments = parse_dbg(xml, books=book_list)
    console.print(f"  • {len(latin_segments)} segmenti latini estratti")

    if italian_jsonl is None:
        console.print("[yellow]⚠[/yellow] Nessun JSONL italiano fornito — "
                      "salto allineamento e embedding TM. "
                      "Per popolare le coppie passa --italian-jsonl PATH")
        return

    italian_segments = [
        ItalianSegment(
            text=row["text"],
            locator=row["locator"],
            translator=row.get("translator", "anonimo"),
            license=row.get("license", "PD"),
        )
        for row in (json.loads(line) for line in italian_jsonl.read_text(encoding="utf-8").splitlines() if line.strip())
    ]
    console.print(f"  • {len(italian_segments)} segmenti italiani caricati")

    pairs = align_by_locator(latin_segments, italian_segments)
    console.print(f"  • {len(pairs)} coppie allineate")

    if not pairs:
        return

    console.print("  • Calcolo embeddings (bge-m3)...")
    vectors = embed([p.source_text for p in pairs])

    rows = [
        {
            "source_text": p.source_text,
            "target_text": p.target_text,
            "author": "Caesar",
            "work": "De Bello Gallico",
            "locator": p.locator,
            "translator": p.translator,
            "license": p.license,
            "embedding": v,
        }
        for p, v in zip(pairs, vectors, strict=True)
    ]
    with session_scope() as session:
        session.execute(insert(TranslationPair), rows)
    console.print(f"[green]✓[/green] Inserite {len(rows)} coppie in translation_pairs")


if __name__ == "__main__":
    cli()
