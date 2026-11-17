"""Inspect the local package without network access."""

import typer

from omni_eval import __version__
from omni_eval.api import metrics
from omni_eval.plugins.catalog import Catalog


def doctor(dry_run: bool = False) -> None:
    """Inspect core environment, without network access."""
    import sys

    typer.echo(f"OmniEval {__version__} · Python {sys.version.split()[0]}")
    typer.echo("DEMO / SYNTHETIC (no endpoints checked)" if dry_run else "Core import: OK")
    typer.echo(f"Native metrics: {len(metrics())} · Planned/ready plugins: {len(Catalog().list())}")
