"""Describe native metric capabilities and prerequisites."""

import json

import typer

from omni_eval.api import metrics
from omni_eval.cli.errors import fail as _failure
from omni_eval.core.registry import Registry


def metric_list(task: str | None = None, dry_run: bool = False) -> None:
    for name in metrics(task):
        typer.echo(name)


def metric_show(name: str, dry_run: bool = False) -> None:
    try:
        typer.echo(json.dumps(Registry().describe(name), indent=2))
    except Exception as exc:
        _failure(exc)
