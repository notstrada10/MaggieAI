"""Phenomena coverage report.

Run every grammar rule's matcher against every `translation_pairs` row
and tally how often each phenomenon fires. Surfaces which rules earn
their slot vs which never match the corpus they're meant to describe.

Live (count > 0) and dead (count == 0) rules are reported separately so
the dead-rule list is unmissable.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
import httpx
from rich.console import Console
from rich.table import Table
from sqlalchemy import select

from maggieai.config import get_settings
from maggieai.db.engine import session_scope
from maggieai.db.models import GrammarRule, TranslationPair
from maggieai.morphology.phenomena import detect
from maggieai.morphology.pipeline import SentenceAnalysis

console = Console()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pair:
    id: int
    source_text: str
    author: str | None
    work: str | None
    locator: str | None


@dataclass
class CoverageReport:
    pairs_analyzed: int
    errors: int
    counts: dict[str, int]
    rules_seen: list[str]
    sample_hits: dict[str, list[int]] = field(default_factory=dict)


def compute_coverage(
    items: Iterable[tuple[Pair, SentenceAnalysis | None]],
    rules: list[dict[str, Any]],
    sample_per_rule: int = 3,
) -> CoverageReport:
    """Pure tally over (pair, analysis) items. `None` analysis means the
    morphology call failed for that pair — counted in `errors`, skipped.
    """
    counts: Counter[str] = Counter()
    samples: dict[str, list[int]] = {}
    rules_seen = sorted({r["phenomenon"] for r in rules})
    errors = 0
    analyzed = 0
    for pair, analysis in items:
        if analysis is None:
            errors += 1
            continue
        analyzed += 1
        for phen in detect(analysis, rules):
            counts[phen] += 1
            bucket = samples.setdefault(phen, [])
            if len(bucket) < sample_per_rule:
                bucket.append(pair.id)
    return CoverageReport(
        pairs_analyzed=analyzed,
        errors=errors,
        counts=dict(counts),
        rules_seen=rules_seen,
        sample_hits=samples,
    )


def _load_pairs(author: str | None, work: str | None, limit: int | None) -> list[Pair]:
    with session_scope() as session:
        stmt = select(TranslationPair).order_by(TranslationPair.id)
        if author:
            stmt = stmt.where(TranslationPair.author == author)
        if work:
            stmt = stmt.where(TranslationPair.work == work)
        if limit:
            stmt = stmt.limit(limit)
        rows = session.scalars(stmt).all()
        return [
            Pair(
                id=r.id,
                source_text=r.source_text,
                author=r.author,
                work=r.work,
                locator=r.locator,
            )
            for r in rows
        ]


def _load_rules() -> list[dict[str, Any]]:
    with session_scope() as session:
        rows = session.scalars(select(GrammarRule)).all()
        return [
            {"phenomenon": r.phenomenon, "pattern": r.pattern, "description": r.description}
            for r in rows
        ]


async def _analyze_all(
    pairs: list[Pair],
    morphology_url: str,
    concurrency: int,
    timeout: float,
) -> list[tuple[Pair, SentenceAnalysis | None]]:
    sem = asyncio.Semaphore(concurrency)
    results: dict[int, SentenceAnalysis | None] = {}

    async with httpx.AsyncClient(base_url=morphology_url, timeout=timeout) as client:

        async def _one(pair: Pair) -> None:
            async with sem:
                try:
                    resp = await client.post("/analyze", json={"text": pair.source_text})
                    resp.raise_for_status()
                    results[pair.id] = SentenceAnalysis.model_validate(resp.json())
                except Exception as exc:
                    logger.warning("morphology error on pair %d: %s", pair.id, exc)
                    results[pair.id] = None

        batch = 20
        for i in range(0, len(pairs), batch):
            chunk = pairs[i : i + batch]
            await asyncio.gather(*(_one(p) for p in chunk))
            console.print(f"  analyzed {min(i + batch, len(pairs))}/{len(pairs)}")

    return [(p, results.get(p.id)) for p in pairs]


def _render(report: CoverageReport) -> None:
    live = sorted(report.counts.items(), key=lambda kv: (-kv[1], kv[0]))
    dead = [p for p in report.rules_seen if p not in report.counts]

    summary = Table(title="Phenomena coverage — summary", show_header=False)
    summary.add_row("Pairs analyzed", str(report.pairs_analyzed))
    summary.add_row("Pairs failed", str(report.errors))
    summary.add_row("Rules total", str(len(report.rules_seen)))
    summary.add_row("Rules live", str(len(live)))
    summary.add_row("Rules dead", str(len(dead)))
    if report.rules_seen:
        summary.add_row("Live %", f"{100 * len(live) / len(report.rules_seen):.1f}%")
    console.print(summary)

    if live:
        live_table = Table(title="Live rules (descending hits)")
        live_table.add_column("Rank", justify="right")
        live_table.add_column("Phenomenon")
        live_table.add_column("Hits", justify="right")
        live_table.add_column("Sample pair_ids")
        for i, (phen, n) in enumerate(live, 1):
            sample = ", ".join(str(x) for x in report.sample_hits.get(phen, []))
            live_table.add_row(str(i), phen, str(n), sample)
        console.print(live_table)

    if dead:
        dead_table = Table(title="Dead rules (zero hits)", show_header=False)
        for phen in dead:
            dead_table.add_row(phen)
        console.print(dead_table)


@click.command("coverage")
@click.option("--morphology-url", default=None, help="Override MORPHOLOGY_URL.")
@click.option("--author", default=None, help="Filter translation_pairs by author.")
@click.option("--work", default=None, help="Filter translation_pairs by work.")
@click.option("--limit", type=int, default=None, help="Truncate to first N pairs.")
@click.option("--concurrency", type=int, default=4, show_default=True)
@click.option(
    "--timeout", type=float, default=60.0, show_default=True, help="Per-call timeout (s)."
)
@click.option(
    "--json-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional JSON dump of the full report.",
)
def coverage(
    morphology_url: str | None,
    author: str | None,
    work: str | None,
    limit: int | None,
    concurrency: int,
    timeout: float,
    json_out: Path | None,
) -> None:
    """Report which grammar rules fire on the corpus and which are dead.

    Runs every rule's UD pattern against each row in `translation_pairs`.
    Useful sanity check after adding rules: are the new ones earning a
    slot, or do their patterns never match real Latin?
    """
    settings = get_settings()
    base_url = morphology_url or settings.morphology_url

    rules = _load_rules()
    if not rules:
        raise click.ClickException("No rules in `grammar_rules`. Did you run `load-grammar`?")

    pairs = _load_pairs(author=author, work=work, limit=limit)
    if not pairs:
        raise click.ClickException("No matching translation_pairs found.")

    console.print(
        f"[bold]Coverage[/bold]: {len(pairs)} pairs x {len(rules)} rules via {base_url}"
    )
    console.print("[dim]First call may take 1-2 min if CLTK has not warmed up.[/dim]")

    items = asyncio.run(
        _analyze_all(pairs, base_url, concurrency=concurrency, timeout=timeout)
    )
    report = compute_coverage(items, rules)
    _render(report)

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(
                {
                    "pairs_analyzed": report.pairs_analyzed,
                    "errors": report.errors,
                    "counts": report.counts,
                    "rules_seen": report.rules_seen,
                    "sample_hits": report.sample_hits,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        console.print(f"[green]Wrote[/green] JSON → {json_out}")
