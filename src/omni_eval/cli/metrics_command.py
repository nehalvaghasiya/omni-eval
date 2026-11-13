import typer

from omni_eval.core.metric_registry import MetricRegistry

metrics_app = typer.Typer(
    no_args_is_help=True, help="Native metric catalog and prerequisites"
)


@metrics_app.command("list")
def list_metrics(dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    if dry_run:
        typer.echo("DRY RUN / SYNTHETIC compatibility preview")

    for descriptor in MetricRegistry().list():
        typer.echo(
            f"{descriptor.id:24} {descriptor.family:16} ref={descriptor.required_reference or 'none':20} prediction={descriptor.required_prediction}"
        )


@metrics_app.command("inspect")
def inspect_metric(
    name: str = typer.Argument("accuracy"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    descriptor = MetricRegistry().get(name if not dry_run else "accuracy")

    if dry_run:
        typer.echo("DRY RUN / SYNTHETIC")

    typer.echo(descriptor)
