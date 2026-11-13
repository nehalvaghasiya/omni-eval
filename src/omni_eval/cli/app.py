"""CLI registration only. All commands call the shared Python API."""

import typer

from omni_eval.cli.compare_command import compare_command
from omni_eval.cli.doctor_command import doctor_command
from omni_eval.cli.evaluate_command import evaluate_command
from omni_eval.cli.init_command import init_command
from omni_eval.cli.judge_command import judge_command
from omni_eval.cli.metrics_command import metrics_app
from omni_eval.cli.replay_command import replay_command
from omni_eval.cli.report_command import report_command
from omni_eval.cli.runner_command import runner_command
from omni_eval.cli.serve_command import serve_command
from omni_eval.cli.validate_command import validate_command

app = typer.Typer(
    help="OmniEval — reproducible evaluation of saved and live predictions",
    no_args_is_help=True,
    invoke_without_command=True,
    add_completion=False,
)
app.command("init")(init_command)
app.command("validate")(validate_command)
app.command("evaluate")(evaluate_command)
app.command("compare")(compare_command)
app.command("replay")(replay_command)
app.command("report")(report_command)
app.command("doctor")(doctor_command)
app.command("judge")(judge_command)
app.command("runner")(runner_command)
app.command("serve")(serve_command)
app.add_typer(metrics_app, name="metrics")


@app.callback()
def root(
    version: bool = typer.Option(False, "--version", help="Display installed version"),
) -> None:
    if version:
        from omni_eval import __version__

        typer.echo(__version__)

        raise typer.Exit()


def main() -> None:
    from pydantic import ValidationError

    from omni_eval.exceptions import (
        AcquisitionError,
        DependencyUnavailable,
        InputError,
        SecurityError,
    )

    try:
        app()
    except (InputError, ValidationError, ValueError) as exc:
        typer.echo(f"invalid input: {exc}", err=True)

        raise SystemExit(2) from exc
    except SecurityError as exc:
        typer.echo(f"security blocked: {exc}", err=True)

        raise SystemExit(5) from exc
    except (AcquisitionError, DependencyUnavailable) as exc:
        typer.echo(f"unavailable: {exc}", err=True)

        raise SystemExit(3) from exc
