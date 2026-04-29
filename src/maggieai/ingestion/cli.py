"""Ingestion CLI (alias `maggie-ingest`).

Commands:
    maggie-ingest init-db                     # apply schema.sql
    maggie-ingest load-grammar [PATH]         # load YAML rules
    maggie-ingest load-corpus dbg --books 1,2 # fetch and ingest DBG
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
@click.option("-v", "--verbose", is_flag=True, help="DEBUG-level logging")
def cli(verbose: bool) -> None:
    """MaggieAI data ingestion."""
    _setup_logging(verbose)


@cli.command("init-db")
def init_db() -> None:
    """Apply `schema.sql` to the configured database."""
    from sqlalchemy import text  # noqa: F401  (kept for psycopg multi-statement script)

    from maggieai.db.engine import get_engine

    schema_path = Path(__file__).parents[1] / "db" / "schema.sql"
    sql = schema_path.read_text(encoding="utf-8")
    with get_engine().begin() as conn:
        # psycopg accepts multi-statement scripts when passed this way
        conn.exec_driver_sql(sql)
    console.print(f"[green]OK[/green] Schema applied from {schema_path}")


@cli.command("load-grammar")
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/grammar_rules"),
)
def load_grammar(path: Path) -> None:
    """Load the YAML rules into `grammar_rules`."""
    from maggieai.ingestion.grammar_loader import load_directory

    n = load_directory(path)
    console.print(f"[green]OK[/green] Loaded {n} grammar rules")


@cli.command("load-corpus")
@click.argument("corpus", type=click.Choice(["dbg"]))
@click.option(
    "--books",
    default="1,2",
    help="CSV of book numbers (e.g. '1,2,3')",
)
@click.option(
    "--translation-jsonl",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="JSONL with the target translation (locator+text+translator). "
    "If omitted, ingest only the Latin text without aligned pairs.",
)
def load_corpus(corpus: str, books: str, translation_jsonl: Path | None) -> None:
    """Fetch a corpus, align with the translation, and populate the TM."""
    import json

    from sqlalchemy import insert

    from maggieai.db.engine import session_scope
    from maggieai.db.models import TranslationPair
    from maggieai.ingestion.aligner import TranslationSegment, align_by_locator
    from maggieai.ingestion.embedder import embed
    from maggieai.ingestion.perseus import fetch_dbg_xml, parse_dbg

    if corpus != "dbg":
        raise click.ClickException(f"Unknown corpus: {corpus}")

    book_list = [int(b.strip()) for b in books.split(",") if b.strip()]
    console.print(f"[bold]Ingesting DBG, books {book_list}[/bold]")

    xml = fetch_dbg_xml()
    latin_segments = parse_dbg(xml, books=book_list)
    console.print(f"  - {len(latin_segments)} Latin segments extracted")

    if translation_jsonl is None:
        console.print(
            "[yellow]![/yellow] No translation JSONL provided — "
            "skipping alignment and TM embedding. "
            "To populate the pairs pass --translation-jsonl PATH"
        )
        return

    translation_segments = [
        TranslationSegment(
            text=row["text"],
            locator=row["locator"],
            translator=row.get("translator", "anonymous"),
            license=row.get("license", "PD"),
        )
        for row in (
            json.loads(line)
            for line in translation_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    ]
    console.print(f"  - {len(translation_segments)} translation segments loaded")

    pairs = align_by_locator(latin_segments, translation_segments)
    console.print(f"  - {len(pairs)} aligned pairs")

    if not pairs:
        return

    console.print("  - Computing embeddings (bge-m3)...")
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
    console.print(f"[green]OK[/green] Inserted {len(rows)} pairs into translation_pairs")


if __name__ == "__main__":
    cli()
