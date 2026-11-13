from pathlib import Path

import typer

from omni_eval.core.capability_registry import preflight
from omni_eval.core.sample_aligner import SampleAligner
from omni_eval.fixtures.fixture_predictions import classification_fixture
from omni_eval.schemas.eval_sample import EvalSample
from omni_eval.schemas.metric_spec import MetricSpec
from omni_eval.schemas.prediction_record import PredictionRecord
from omni_eval.sources.jsonl_loader import load_jsonl


def validate_command(
    samples: Path | None = typer.Option(None, "--samples"),
    predictions: Path | None = typer.Option(None, "--predictions"),
    metrics: str = typer.Option("accuracy", "--metrics"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    if dry_run:
        records, outputs = classification_fixture()
    else:
        if samples is None or predictions is None:
            raise ValueError("validate requires --samples and --predictions")

        records = load_jsonl(samples, EvalSample)
        outputs = load_jsonl(predictions, PredictionRecord)

    selected = [
        MetricSpec(name=metric.strip())
        for metric in metrics.split(",")
        if metric.strip()
    ]
    preflight(records, selected)
    aligned, extras = SampleAligner().align(records, outputs)
    missing = [s.sample_id for s, p in aligned if p is None]
    typer.echo(
        f"{'DRY RUN / SYNTHETIC: ' if dry_run else ''}validated={len(records)} missing={len(missing)} extra={len(extras)}"
    )

    if missing or extras:
        raise ValueError(f"alignment incomplete: missing={missing}; extra={extras}")
