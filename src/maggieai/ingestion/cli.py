"""Ingestion CLI (alias `maggie-ingest`).

Commands:
    maggie-ingest init-db                     # apply schema.sql
    maggie-ingest load-grammar [PATH]         # load YAML rules
    maggie-ingest load-corpus dbg --books 1,2 # fetch and ingest DBG
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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


@cli.command("fetch-translation")
@click.argument("corpus")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path for the JSONL output. Defaults to data/raw/{corpus}-eng.jsonl.",
)
@click.option(
    "--books",
    default=None,
    help="CSV of book numbers to include. Defaults to all books of the work.",
)
def fetch_translation(corpus: str, output: Path | None, books: str | None) -> None:
    """Fetch a public-domain English translation as a JSONL.

    CORPUS is the slug of a Perseus work registered in PERSEUS_WORKS
    (currently ``dbg`` for De Bello Gallico, ``dbc`` for De Bello
    Civili — both Caesar). Output is one JSONL record per chapter at
    chapter-level locator (book.chapter), matching what the matching
    :command:`load-corpus` command expects.
    """
    import json

    from maggieai.ingestion.perseus import PERSEUS_WORKS, fetch_dbg_xml, parse_dbg

    work = PERSEUS_WORKS.get(corpus)
    if work is None:
        choices = ", ".join(sorted(PERSEUS_WORKS))
        raise click.ClickException(f"Unknown corpus '{corpus}'. Choices: {choices}")

    output_path = output if output is not None else Path(f"data/raw/{corpus}-eng.jsonl")
    book_list = (
        [int(b.strip()) for b in books.split(",") if b.strip()] if books else None
    )
    label = f"books {book_list}" if book_list else "all books"
    console.print(
        f"[bold]Fetching {work.author} — {work.work} English translation, {label}[/bold]"
    )

    xml = fetch_dbg_xml(url=work.eng_url)
    segments = parse_dbg(xml, books=book_list, granularity="chapter")
    console.print(f"  - {len(segments)} chapter segments extracted")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            record = {
                "locator": seg.locator,
                "text": seg.text,
                "translator": work.eng_translator,
                "license": work.license,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    console.print(f"[green]OK[/green] Wrote {output_path}")


def _detect_granularity(translation_segments: list[Any]) -> str:
    """A locator like '1.1.1' is section-level; '1.1' is chapter-level."""
    if not translation_segments:
        return "section"
    sample = translation_segments[0].locator
    return "chapter" if sample.count(".") == 1 else "section"


@cli.command("load-corpus")
@click.argument("corpus")
@click.option(
    "--books",
    default=None,
    help="CSV of book numbers. Defaults to all books of the work.",
)
@click.option(
    "--translation-jsonl",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="JSONL with the target translation (locator+text+translator). "
    "If omitted, ingest only the Latin text without aligned pairs.",
)
def load_corpus(corpus: str, books: str | None, translation_jsonl: Path | None) -> None:
    """Fetch a corpus, align with the translation, populate the TM.

    Idempotent: re-running with the same `(author, work, locator)` keys
    upserts the row (handy for editing translations or re-embedding).

    Latin granularity is auto-derived from the translation JSONL's
    locator depth — `1.1` is chapter-level, `1.1.1` is section-level.
    """
    import json

    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from maggieai.db.engine import session_scope
    from maggieai.db.models import TranslationPair
    from maggieai.ingestion.aligner import TranslationSegment, align_by_locator
    from maggieai.ingestion.embedder import embed
    from maggieai.ingestion.perseus import PERSEUS_WORKS, fetch_dbg_xml, parse_dbg

    work = PERSEUS_WORKS.get(corpus)
    if work is None:
        choices = ", ".join(sorted(PERSEUS_WORKS))
        raise click.ClickException(f"Unknown corpus '{corpus}'. Choices: {choices}")

    book_list = (
        [int(b.strip()) for b in books.split(",") if b.strip()] if books else None
    )
    label = f"books {book_list}" if book_list else "all books"
    console.print(f"[bold]Ingesting {work.author} — {work.work}, {label}[/bold]")

    if translation_jsonl is None:
        xml = fetch_dbg_xml(url=work.lat_url)
        latin_segments = parse_dbg(xml, books=book_list)
        console.print(f"  - {len(latin_segments)} Latin segments extracted")
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
    granularity = _detect_granularity(translation_segments)
    console.print(
        f"  - {len(translation_segments)} translation segments loaded "
        f"(granularity={granularity})"
    )

    xml = fetch_dbg_xml(url=work.lat_url)
    latin_segments = parse_dbg(xml, books=book_list, granularity=granularity)  # type: ignore[arg-type]
    console.print(f"  - {len(latin_segments)} Latin segments extracted")

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
            "author": work.author,
            "work": work.work,
            "locator": p.locator,
            "translator": p.translator,
            "license": p.license,
            "embedding": v,
        }
        for p, v in zip(pairs, vectors, strict=True)
    ]
    with session_scope() as session:
        # Idempotent upsert on (author, work, locator). On collision we
        # refresh the text + embedding (translator can be edited and
        # re-embedded without touching the row identity).
        stmt = pg_insert(TranslationPair).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="translation_pairs_author_work_locator_uq",
            set_={
                "source_text": stmt.excluded.source_text,
                "target_text": stmt.excluded.target_text,
                "translator": stmt.excluded.translator,
                "license": stmt.excluded.license,
                "embedding": stmt.excluded.embedding,
            },
        )
        session.execute(stmt)
    console.print(
        f"[green]OK[/green] Upserted {len(rows)} pairs into translation_pairs"
    )


if __name__ == "__main__":
    cli()
