from pathlib import Path

import typer

from omni_eval.adapters.runner_result_importer import RunnerResultImporter
from omni_eval.runners.fixture_runner import FixtureRunner


def runner_command(
    result: Path | None = typer.Option(
        None, "--result", help="Import a completed external runner result JSON"
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    if dry_run:
        manifest = FixtureRunner().execute()
    else:
        if result is None:
            raise ValueError(
                "runner requires --result (external execution is not enabled)"
            )

        manifest = RunnerResultImporter().import_file(result)

    typer.echo(
        f"{'DRY RUN / SYNTHETIC — ' if manifest.is_fixture else ''}runner={manifest.runner_id} version={manifest.runner_version} status={manifest.status}"
    )

    for key, value in manifest.metrics.items():
        typer.echo(f"  {key}: {value}")
