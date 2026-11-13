import tempfile
from pathlib import Path

import typer

from omni_eval.api import evaluate, load_run
from omni_eval.reports.html_reporter import HtmlReporter
from omni_eval.reports.json_reporter import JsonReporter
from omni_eval.reports.markdown_reporter import MarkdownReporter


def report_command(
    run: Path | None = typer.Option(None, "--run"),
    format: str = typer.Option("html", "--format"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    if dry_run:
        with tempfile.TemporaryDirectory(prefix="omni-eval-report-") as temporary:
            result = evaluate(dry_run=True, output_dir=temporary)
            typer.echo(
                "DRY RUN / SYNTHETIC: report rendered and validated in temporary storage"
            )

            if "DRY RUN" not in HtmlReporter().render(result):
                raise ValueError("fixture marker missing")

        return

    if run is None:
        raise ValueError("report requires --run")

    result = load_run(run)
    renderers: dict[
        str, type[HtmlReporter] | type[JsonReporter] | type[MarkdownReporter]
    ] = {
        "html": HtmlReporter,
        "json": JsonReporter,
        "markdown": MarkdownReporter,
    }

    if format not in renderers:
        raise ValueError("supported formats: html,json,markdown")

    typer.echo(renderers[format]().render(result))
