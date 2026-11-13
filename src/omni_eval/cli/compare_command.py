from pathlib import Path

import typer

from omni_eval.api import compare, evaluate
from omni_eval.fixtures.fixture_predictions import classification_fixture


def compare_command(
    baseline: Path | None = typer.Option(None, "--baseline"),
    candidate: Path | None = typer.Option(None, "--candidate"),
    paired: bool = typer.Option(True, "--paired/--unpaired"),
    confidence: float = typer.Option(0.95, "--confidence"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    gate_metric: str | None = typer.Option(None, "--gate-metric"),
    gate_min_delta: float | None = typer.Option(None, "--gate-min-delta"),
    fail_on_inconclusive: bool = typer.Option(False, "--fail-on-inconclusive"),
) -> None:
    if dry_run:
        from omni_eval.schemas.prediction_record import PredictionRecord

        samples, predictions = classification_fixture()
        improved = [
            PredictionRecord.model_validate(p.model_dump(mode="json"))
            for p in predictions
        ]
        improved[2].output["label"] = "yes"
        left = evaluate(samples, predictions, ["accuracy", "macro_f1"], dry_run=False)
        right = evaluate(samples, improved, ["accuracy", "macro_f1"], dry_run=False)
        left.manifest.is_fixture = right.manifest.is_fixture = True
    else:
        if baseline is None or candidate is None:
            raise ValueError("compare requires --baseline and --candidate")

        left, right = baseline, candidate

    typer.echo(
        "DRY RUN / SYNTHETIC — NOT A BENCHMARK" if dry_run else "Paired comparison"
    )
    results = compare(
        left,
        right,
        paired=paired,
        confidence=confidence,
        iterations=100 if dry_run else 1000,
    )

    for result in results:
        typer.echo(
            f"{result.metric}: delta={result.delta} paired_n={result.paired_n} CI={result.interval} status={result.status}"
        )

    if gate_metric:
        matched = next(
            (result for result in results if result.metric == gate_metric), None
        )

        if matched is None or matched.delta is None or matched.status != "completed":
            typer.echo(f"gate {gate_metric}: inconclusive")

            if fail_on_inconclusive:
                raise typer.Exit(4)
        elif gate_min_delta is not None and matched.delta < gate_min_delta:
            typer.echo(f"gate {gate_metric}: fail (delta below threshold)")

            raise typer.Exit(1)
        else:
            typer.echo(f"gate {gate_metric}: pass")
