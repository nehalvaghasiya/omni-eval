"""One-command evaluation through the public API."""

from pathlib import Path
from typing import Annotated

import typer

from omni_eval.api import evaluate
from omni_eval.cli.errors import fail as _failure
from omni_eval.cli.output import print_run as _print_run
from omni_eval.types.config import Config


def run(
    data: Annotated[Path | None, typer.Argument(help="JSONL cases file")] = None,
    task: Annotated[
        str | None, typer.Option("--task", help="text, retrieval, rag, agent, classification, ...")
    ] = None,
    config: Annotated[
        Path | None, typer.Option("-c", "--config", help="YAML evaluation config")
    ] = None,
    metric: Annotated[
        list[str] | None, typer.Option("--metric", help="Repeat to select metric names")
    ] = None,
    out: Annotated[Path | None, typer.Option("--out", help="Runs directory")] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Use genuine scoring on synthetic cases only")
    ] = False,
) -> None:
    """Evaluate saved data, configured endpoint, or selected plugin."""
    try:
        spec: Config | Path
        if config:
            spec = Config.from_file(config)
        elif data:
            spec = data
        elif dry_run:
            spec = Config(task=task or "classification", data="synthetic")
        else:
            raise ValueError("provide a cases file or -c config; see 'omni-eval run --help'")
        _print_run(evaluate(spec, task=task, metrics=metric or None, out=out, dry_run=dry_run))
    except Exception as exc:
        _failure(exc)
