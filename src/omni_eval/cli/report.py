"""Present saved evaluation results without rerunning scorers."""

import json
from pathlib import Path
from typing import Annotated

import typer

from omni_eval.api import evaluate, load
from omni_eval.cli.errors import fail as _failure
from omni_eval.fixtures.data import cases as get_demo
from omni_eval.reports.html import Html
from omni_eval.reports.json import Json
from omni_eval.reports.junit import Junit
from omni_eval.reports.md import Markdown


def report(
    run: Annotated[Path, typer.Argument()] = Path("run"),
    format: str = "html",
    dry_run: bool = False,
) -> None:
    """Render a saved run; no scorer is recomputed."""
    try:
        result = (
            evaluate(get_demo("classification"), task="classification", dry_run=True)
            if dry_run
            else load(run)
        )
        output: dict[str, Html | Json | Markdown | Junit] = {
            "html": Html(),
            "json": Json(),
            "md": Markdown(),
            "junit": Junit(),
        }
        if format not in output:
            raise ValueError("format must be html, json, md or junit")
        produced = output[format].render(result)
        typer.echo(json.dumps(produced, indent=2) if isinstance(produced, dict) else produced)
    except Exception as exc:
        _failure(exc)
