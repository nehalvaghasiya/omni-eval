"""CI gate command for saved runs."""

import json
from pathlib import Path
from typing import Annotated

import typer

from omni_eval.api import evaluate, load
from omni_eval.cli.errors import fail as _failure
from omni_eval.core.gates import GateEngine
from omni_eval.fixtures.data import cases as get_demo
from omni_eval.types.gate import GateRule


def check(
    run: Annotated[Path, typer.Argument()] = Path("run"),
    config: Annotated[Path | None, typer.Option("-c", "--config")] = None,
    dry_run: bool = False,
    strict: Annotated[bool, typer.Option("--strict", help="Exit 4 for inconclusive gates")] = False,
) -> None:
    """Check explicit thresholds with pass/fail/inconclusive semantics."""
    try:
        result = (
            evaluate(get_demo("classification"), task="classification", dry_run=True)
            if dry_run
            else load(run)
        )
        if config:
            from omni_eval.io.yaml import load as yaml_load

            rules = [GateRule.model_validate(row) for row in yaml_load(config).get("gates", [])]
        else:
            rules = [GateRule(metric="accuracy", value=0.5, min_coverage=1)]
        outcomes = GateEngine().check(result, rules)
        typer.echo(json.dumps([x.model_dump() for x in outcomes], indent=2))
        if any(x.status == "fail" for x in outcomes):
            raise typer.Exit(code=1)
        if strict and any(x.status == "inconclusive" for x in outcomes):
            raise typer.Exit(code=4)
    except typer.Exit:
        raise
    except Exception as exc:
        _failure(exc)
