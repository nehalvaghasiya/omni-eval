"""Uniform, transparent CLI result printing."""

import typer

from omni_eval.types.run import Run


def print_run(run: object) -> None:
    assert isinstance(run, Run)
    typer.echo("DEMO / SYNTHETIC" if run.fixture else f"OmniEval run {run.id}")
    for score in run.scores:
        result = f"{score.value:.6g}" if score.value is not None else "unavailable"
        typer.echo(f"  {score.name}: {result} [{score.status}, coverage {score.coverage:.0%}]")
    if run.path:
        typer.echo(f"  Saved: {run.path}")
