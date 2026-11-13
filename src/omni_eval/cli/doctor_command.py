import importlib.util
import platform

import typer


def doctor_command(dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    if dry_run:
        typer.echo(
            "DRY RUN / SYNTHETIC: core fixture environment healthy; optional runners unavailable in this simulation"
        )

        return

    typer.echo(f"Python {platform.python_version()}")

    for package in (
        "pydantic",
        "numpy",
        "sklearn",
        "httpx",
        "deepeval",
        "ragas",
        "fastapi",
        "pyarrow",
    ):
        typer.echo(
            f"{package}: {'available' if importlib.util.find_spec(package) else 'not installed (optional if not core)'}"
        )
