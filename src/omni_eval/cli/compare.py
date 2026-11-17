"""Paired comparisons of two saved run artifacts."""

import json
from pathlib import Path
from typing import Annotated

import typer

from omni_eval.api import compare, evaluate, load
from omni_eval.cli.errors import fail as _failure
from omni_eval.fixtures.data import cases as get_demo


def compare_runs(
    baseline: Path,
    candidate: Path,
    metric: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    """Paired sample comparison and confidence interval."""
    try:
        if dry_run:
            old = evaluate(get_demo("classification"), task="classification", dry_run=True)
            changed = [
                c.model_copy(update={"actual": {"label": "A"}}) if c.id == "c2" else c
                for c in get_demo("classification")
            ]
            new = evaluate(changed, task="classification", metrics=["accuracy", "macro_f1"])
        else:
            old = load(baseline)
            new = load(candidate)
        typer.echo(json.dumps(compare(old, new, metrics=metric or None).summary, indent=2))
    except Exception as exc:
        _failure(exc)


def compare_command(
    baseline: Annotated[Path, typer.Argument()] = Path("baseline"),
    candidate: Annotated[Path, typer.Argument()] = Path("candidate"),
    metric: Annotated[list[str] | None, typer.Option("--metric")] = None,
    dry_run: bool = False,
) -> None:
    compare_runs(baseline, candidate, metric, dry_run)
