"""Create a minimal runnable evaluation workspace."""

from pathlib import Path
from typing import Annotated

import typer

from omni_eval.cli.errors import fail as _failure
from omni_eval.fixtures.data import cases as get_demo


def init(
    path: Annotated[Path, typer.Argument()] = Path("."),
    task: str = "classification",
    dry_run: bool = False,
) -> None:
    """Create one beginner YAML and one example cases file."""
    try:
        cases = get_demo(task)
        files = [path / "cases.jsonl", path / "eval.yaml"]
        for file in files:
            if file.exists():
                raise FileExistsError(f"will not overwrite {file}")
        typer.echo("Would create: " + ", ".join(str(f) for f in files))
        if dry_run:
            return
        path.mkdir(parents=True, exist_ok=True)
        from omni_eval.io.jsonl import Jsonl

        Jsonl().write(files[0], [c.model_dump(mode="json") for c in cases])
        metric = {"security": "attack_success", "code": "code_valid", "sql": "sql_select_safe"}.get(
            task
        )
        metrics = f"metrics: [{metric}]\n" if metric else ""
        files[1].write_text(
            f"version: 1\ntask: {task}\ndata: cases.jsonl\n{metrics}", encoding="utf-8"
        )
    except Exception as exc:
        _failure(exc)
