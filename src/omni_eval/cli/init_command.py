from pathlib import Path

import typer

from omni_eval.fixtures.fixture_predictions import classification_fixture
from omni_eval.sources.jsonl_loader import write_jsonl


def init_command(
    directory: Path = typer.Option(Path("omni-eval-example"), "--directory"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    files = ("samples.jsonl", "predictions.jsonl", "config.yaml")

    if dry_run:
        typer.echo(
            "DRY RUN / SYNTHETIC: would create "
            + ", ".join(str(directory / name) for name in files)
        )

        return

    if any((directory / file).exists() for file in files):
        raise ValueError("init refuses to overwrite existing example files")

    directory.mkdir(parents=True, exist_ok=True)
    samples, predictions = classification_fixture()
    write_jsonl(directory / "samples.jsonl", samples)
    write_jsonl(directory / "predictions.jsonl", predictions)
    (directory / "config.yaml").write_text(
        "schema_version: '1.0'\nname: my-eval\ndataset: samples.jsonl\nsource:\n  mode: offline\n  predictions: predictions.jsonl\nmetrics:\n  - name: accuracy\n  - name: macro_f1\nreport:\n  formats: [json, html]\n",
        encoding="utf-8",
    )
    typer.echo(f"example created in {directory} (synthetic fixture data)")
