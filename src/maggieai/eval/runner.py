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

from maggieai.eval.coverage import coverage as _coverage_cmd

console = Console()
logger = logging.getLogger(__name__)

RESPONDEO_TRANSLATION_URL = (
    "https://raw.githubusercontent.com/slanglab/RespondeoQA/main/"
    "data/final_dataset/junior_scholarship_translation_long.json"
)


@click.group()
def cli() -> None:
    """MaggieAI eval CLI."""


cli.add_command(_coverage_cmd)


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
        r"^(Derive and translate|Translate)[\s—–\-:]+",  # noqa: RUF001 - en/em dashes appear verbatim in RespondeoQA prompts
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
@click.option(
    "--routing-mode",
    type=click.Choice(["hybrid", "claude-only", "local-only", "deepseek-only"]),
    default=None,
    help="Override gateway routing for this run (sent as routing_mode in /translate body).",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write per-item results (input, reference, hypothesis, error) to this JSONL.",
)
def run(
    gold: Path,
    gateway_url: str,
    limit: int | None,
    routing_mode: str | None,
    output: Path | None,
) -> None:
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
    if routing_mode:
        console.print(f"[dim]routing_mode={routing_mode}[/dim]")
    hyp: list[str] = []
    ref: list[str] = []
    per_item: list[dict[str, object]] = []

    async def _run_one(client: httpx.AsyncClient, text: str) -> str:
        payload: dict[str, object] = {"text": text}
        if routing_mode:
            payload["routing_mode"] = routing_mode
        r = await client.post(f"{gateway_url}/translate", json=payload, timeout=600.0)
        r.raise_for_status()
        return str(r.json()["translation"])

    async def _drive() -> None:
        async with httpx.AsyncClient() as client:
            for rec in records:
                err: str | None = None
                try:
                    out = await _run_one(client, rec["input"])
                except Exception as exc:
                    logger.warning("Error on input '%s': %s", rec["input"][:50], exc)
                    out = ""
                    err = f"{type(exc).__name__}: {exc}"
                hyp.append(out)
                ref.append(rec["translation"])
                per_item.append(
                    {
                        "input": rec["input"],
                        "reference": rec["translation"],
                        "hypothesis": out,
                        "error": err,
                        "question_id": rec.get("question_id"),
                    }
                )
                console.print(f"  [{len(hyp)}/{len(records)}] {rec['input'][:60]}...")

    asyncio.run(_drive())

    bleu = corpus_bleu(hyp, [ref])
    chrf = corpus_chrf(hyp, [ref])

    table = Table(title="Eval results")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("BLEU", f"{bleu.score:.2f}")
    table.add_row("chrF", f"{chrf.score:.2f}")
    n_errors = sum(1 for it in per_item if it["error"])
    if n_errors:
        table.add_row("Errors", f"{n_errors}/{len(records)}")
    console.print(table)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as fh:
            for item in per_item:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        console.print(f"[green]Wrote[/green] {len(per_item)} per-item rows → {output}")


def _load_run(path: Path) -> list[dict[str, object]]:
    """Read a JSONL produced by ``maggie-eval run --output``."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _align_runs(
    a: list[dict[str, object]], b: list[dict[str, object]]
) -> tuple[list[tuple[dict[str, object], dict[str, object]]], list[str]]:
    """Pair items by question_id (falling back to input).

    Returns (aligned_pairs, warnings). Items present in only one side
    produce a warning string but are dropped from the aligned list — a
    fair comparison needs both hypotheses.
    """

    def _key(item: dict[str, object]) -> str:
        qid = item.get("question_id")
        return str(qid) if qid else str(item.get("input", ""))

    by_key_a = {_key(it): it for it in a}
    by_key_b = {_key(it): it for it in b}
    common = sorted(set(by_key_a) & set(by_key_b))
    only_a = sorted(set(by_key_a) - set(by_key_b))
    only_b = sorted(set(by_key_b) - set(by_key_a))

    warnings: list[str] = []
    if only_a:
        warnings.append(f"{len(only_a)} item(s) only in A (dropped)")
    if only_b:
        warnings.append(f"{len(only_b)} item(s) only in B (dropped)")
    return [(by_key_a[k], by_key_b[k]) for k in common], warnings


def _sentence_bleu(hyp: str, ref: str) -> float:
    """Sentence-level BLEU. Returns 0.0 for empty hypothesis (the error case)."""
    if not hyp:
        return 0.0
    from sacrebleu import sentence_bleu

    return float(sentence_bleu(hyp, [ref]).score)


@cli.command("compare")
@click.argument("a", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("b", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--top", type=int, default=5, show_default=True, help="Top-N wins/losses to print.")
def compare(a: Path, b: Path, top: int) -> None:
    """Diff two JSONL run outputs from `maggie-eval run --output`.

    Shows corpus BLEU/chrF + error counts for each side, the items where
    A and B disagree most by sentence-BLEU delta, and the error-set diff.
    """
    try:
        from sacrebleu import corpus_bleu, corpus_chrf
    except ImportError as exc:  # pragma: no cover
        raise click.ClickException(
            "sacrebleu is not installed. Install with: uv pip install -e '.[eval]'"
        ) from exc

    run_a = _load_run(a)
    run_b = _load_run(b)
    aligned, align_warnings = _align_runs(run_a, run_b)
    for w in align_warnings:
        console.print(f"[yellow]Warning:[/yellow] {w}")

    if not aligned:
        raise click.ClickException("No items align between A and B (check question_id/input keys).")

    hyp_a = [str(p[0].get("hypothesis", "")) for p in aligned]
    hyp_b = [str(p[1].get("hypothesis", "")) for p in aligned]
    refs = [str(p[0].get("reference", "")) for p in aligned]
    err_a = sum(1 for p in aligned if p[0].get("error"))
    err_b = sum(1 for p in aligned if p[1].get("error"))

    summary = Table(title=f"Compare: {a.name} vs {b.name} ({len(aligned)} aligned items)")
    summary.add_column("Metric")
    summary.add_column("A", justify="right")
    summary.add_column("B", justify="right")
    summary.add_column("Δ (B-A)", justify="right")

    bleu_a = corpus_bleu(hyp_a, [refs]).score
    bleu_b = corpus_bleu(hyp_b, [refs]).score
    chrf_a = corpus_chrf(hyp_a, [refs]).score
    chrf_b = corpus_chrf(hyp_b, [refs]).score
    summary.add_row("BLEU", f"{bleu_a:.2f}", f"{bleu_b:.2f}", f"{bleu_b - bleu_a:+.2f}")
    summary.add_row("chrF", f"{chrf_a:.2f}", f"{chrf_b:.2f}", f"{chrf_b - chrf_a:+.2f}")
    summary.add_row("Errors", f"{err_a}", f"{err_b}", f"{err_b - err_a:+d}")
    console.print(summary)

    # Per-item deltas for the top-K disagreements
    deltas: list[tuple[float, dict[str, object], dict[str, object]]] = []
    for item_a, item_b in aligned:
        ref = str(item_a.get("reference", ""))
        ha = str(item_a.get("hypothesis", ""))
        hb = str(item_b.get("hypothesis", ""))
        delta = _sentence_bleu(hb, ref) - _sentence_bleu(ha, ref)
        deltas.append((delta, item_a, item_b))

    def _print_top(title: str, sorted_deltas: list[tuple[float, dict[str, object], dict[str, object]]]) -> None:
        t = Table(title=title)
        t.add_column("Δ", justify="right")
        t.add_column("Input")
        t.add_column("A")
        t.add_column("B")
        for delta, item_a, item_b in sorted_deltas[:top]:
            t.add_row(
                f"{delta:+.1f}",
                str(item_a.get("input", ""))[:60],
                str(item_a.get("hypothesis", ""))[:60] or "[red]<error>[/red]",
                str(item_b.get("hypothesis", ""))[:60] or "[red]<error>[/red]",
            )
        console.print(t)

    _print_top("Biggest B wins", sorted(deltas, key=lambda d: -d[0]))
    _print_top("Biggest A wins", sorted(deltas, key=lambda d: d[0]))

    # Error-set diff
    errs_only_in_a = [p for p in aligned if p[0].get("error") and not p[1].get("error")]
    errs_only_in_b = [p for p in aligned if p[1].get("error") and not p[0].get("error")]
    if errs_only_in_a or errs_only_in_b:
        console.print(
            f"[dim]Errors only in A: {len(errs_only_in_a)}; "
            f"only in B: {len(errs_only_in_b)}[/dim]"
        )


if __name__ == "__main__":
    cli()
