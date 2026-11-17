"""Typer command registration only; business logic lives in focused modules."""

import typer

from omni_eval import __version__
from omni_eval.cli.check import check
from omni_eval.cli.compare import compare_command
from omni_eval.cli.demo import demo
from omni_eval.cli.doctor import doctor
from omni_eval.cli.init import init
from omni_eval.cli.metrics import metric_list, metric_show
from omni_eval.cli.plugins import plugin_doctor, plugin_list, plugin_setup, plugin_show
from omni_eval.cli.report import report
from omni_eval.cli.run import run

app = typer.Typer(help="OmniEval: simple, reproducible AI evaluations", no_args_is_help=True)
metrics_app = typer.Typer(help="List and inspect native metric formulas")
plugins_app = typer.Typer(help="Inspect and prepare optional evaluation plugins")
app.command("run")(run)
app.command("demo")(demo)
app.command("init")(init)
app.command("compare")(compare_command)
app.command("report")(report)
app.command("check")(check)
app.command("doctor")(doctor)
metrics_app.command("list")(metric_list)
metrics_app.command("show")(metric_show)
plugins_app.command("list")(plugin_list)
plugins_app.command("show")(plugin_show)
plugins_app.command("doctor")(plugin_doctor)
plugins_app.command("setup")(plugin_setup)
app.add_typer(metrics_app, name="metrics")
app.add_typer(plugins_app, name="plugins")


@app.callback(invoke_without_command=True)
def root(version: bool = typer.Option(False, "--version", help="Show package version")) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


def main() -> None:
    app()
