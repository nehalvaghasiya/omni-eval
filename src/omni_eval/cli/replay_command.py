import tempfile
from pathlib import Path

import typer

from omni_eval.api import evaluate, load_run
from omni_eval.cli.evaluate_command import render_summary
from omni_eval.core.evaluation_engine import EvaluationEngine
from omni_eval.schemas.metric_spec import MetricSpec
from omni_eval.storage.file_run_repository import FileRunRepository


def replay_command(
    run: Path | None = typer.Option(None, "--run"),
    metrics: str = typer.Option("accuracy", "--metrics"),
    out: Path = typer.Option(Path(".omni-eval/runs"), "--out"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    if dry_run:
        with tempfile.TemporaryDirectory(prefix="omni-eval-replay-") as temporary:
            first = evaluate(dry_run=True, output_dir=temporary)
            replayed = EvaluationEngine().evaluate(
                first.samples,
                first.predictions,
                [MetricSpec(name="accuracy")],
                name="DRY RUN / SYNTHETIC REPLAY",
                source_mode="replay",
                is_fixture=True,
            )
            FileRunRepository().save(replayed, temporary)
            render_summary(replayed)

        return

    if run is None:
        raise ValueError("replay requires --run")

    saved = load_run(run)

    if not saved.samples or not saved.predictions:
        raise ValueError("replay requires retained samples/predictions")

    metrics_list = [
        MetricSpec(name=name.strip()) for name in metrics.split(",") if name.strip()
    ]
    replayed = EvaluationEngine().evaluate(
        saved.samples,
        saved.predictions,
        metrics_list,
        name=f"replay of {saved.manifest.run_id}",
        source_mode="replay",
        is_fixture=saved.manifest.is_fixture,
    )
    replayed.manifest.metadata["replay_of"] = saved.manifest.run_id
    FileRunRepository().save(replayed, out)
    render_summary(replayed)
