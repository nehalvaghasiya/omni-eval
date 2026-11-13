import tempfile
from pathlib import Path

import typer

from omni_eval.api import evaluate, load_spec
from omni_eval.schemas.run_result import RunResult


def render_summary(run: RunResult) -> None:
    typer.echo(
        ("DRY RUN / SYNTHETIC — NOT A BENCHMARK\n" if run.manifest.is_fixture else "")
        + f"run={run.manifest.run_id} status={run.manifest.status}"
    )

    if run.path and not run.manifest.is_fixture:
        typer.echo(f"artifacts={run.path}")

    for metric in run.metrics:
        value = f"{metric.value:.6g}" if metric.value is not None else "N/A"
        typer.echo(
            f"{metric.name}: {value} ({metric.status}; {metric.n_scored}/{metric.n_total} scored)"
        )


def evaluate_command(
    samples: Path | None = typer.Option(
        None, "--samples", help="Canonical JSONL samples"
    ),
    predictions: Path | None = typer.Option(
        None, "--predictions", help="Saved canonical JSONL predictions"
    ),
    metrics: str = typer.Option(
        "accuracy", "--metrics", help="Comma-separated registered metric IDs"
    ),
    config: Path | None = typer.Option(None, "--config", help="Typed YAML RunSpec"),
    out: Path = typer.Option(Path(".omni-eval/runs"), "--out"),
    strict: bool = typer.Option(False, "--strict"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    gate_metric: str | None = typer.Option(
        None, "--gate-metric", help="Metric ID to gate"
    ),
    gate_min: float | None = typer.Option(
        None, "--gate-min", help="Required minimum metric value"
    ),
    gate_max: float | None = typer.Option(
        None, "--gate-max", help="Required maximum metric value"
    ),
    min_coverage: float = typer.Option(1.0, "--min-coverage"),
    fail_on_inconclusive: bool = typer.Option(False, "--fail-on-inconclusive"),
) -> None:
    if dry_run:
        with tempfile.TemporaryDirectory(prefix="omni-eval-fixture-") as temporary:
            run = evaluate(
                dry_run=True,
                metrics=["accuracy", "macro_f1", "brier_score", "ece"],
                output_dir=temporary,
            )
            render_summary(run)

        return

    if config:
        run = evaluate(load_spec(config), output_dir=out, strict=strict)
    else:
        run = evaluate(
            samples,
            predictions,
            [s.strip() for s in metrics.split(",") if s.strip()],
            output_dir=out,
            strict=strict,
        )

    render_summary(run)

    if gate_metric is not None:
        from omni_eval.comparison.gate_evaluator import GateEvaluator

        result = next((m for m in run.metrics if m.name == gate_metric), None)
        gate = GateEvaluator().evaluate(
            result,
            min_value=gate_min,
            max_value=gate_max,
            min_coverage=min_coverage,
            gate_name=gate_metric,
        )
        typer.echo(f"gate {gate.gate_name}: {gate.status} — {gate.reason}")

        if gate.status == "fail":
            raise typer.Exit(1)

        if gate.status == "inconclusive" and fail_on_inconclusive:
            raise typer.Exit(4)
