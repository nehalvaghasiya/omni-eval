"""Only known synthetic cases; no endpoint/model access."""

from typing import Annotated

import typer

from omni_eval.api import evaluate
from omni_eval.cli.errors import fail as _failure
from omni_eval.cli.output import print_run as _print_run
from omni_eval.fixtures.data import cases as get_demo


def demo(task: Annotated[str, typer.Argument()] = "classification", dry_run: bool = False) -> None:
    """Actual engine on checked, synthetic built-in examples (no network)."""
    try:
        selected = get_demo(task)
        preset = {
            "security": ["attack_success"],
            "code": ["code_valid"],
            "sql": ["sql_select_safe"],
        }.get(task, [])
        _print_run(evaluate(selected, task=task, metrics=preset or None, dry_run=True))
    except Exception as exc:
        _failure(exc)
