"""All native task presets and command dry-runs execute offline."""

import pytest
from typer.testing import CliRunner

from omni_eval.cli.app import app
from omni_eval.fixtures.data import names


def cli(*args: str):
    """Exercise the actual Typer application without costly subprocess startup."""
    return CliRunner().invoke(app, list(args), catch_exceptions=False)


@pytest.mark.parametrize("task", names())
def test_synthetic_demo(task):
    result = cli("demo", task, "--dry-run")
    assert result.exit_code == 0, result.output
    assert "DEMO / SYNTHETIC" in result.stdout


@pytest.mark.parametrize(
    "args",
    [
        ("run", "--dry-run"),
        ("compare", "--dry-run"),
        ("report", "--dry-run"),
        ("check", "--dry-run"),
        ("doctor", "--dry-run"),
        ("init", "--dry-run"),
        ("metrics", "list", "--dry-run"),
        ("metrics", "show", "accuracy", "--dry-run"),
        ("plugins", "list", "--dry-run"),
        ("plugins", "show", "ragas", "--dry-run"),
        ("plugins", "doctor", "ragas", "--dry-run"),
        ("plugins", "setup", "ragas", "--dry-run"),
    ],
)
def test_dry_commands(args, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = cli(*args)
    assert result.exit_code == 0, (args, result.output)


def test_cli_help_and_version():
    assert cli("--help").exit_code == 0
    assert cli("--version").exit_code == 0


def test_invalid_input_exit_code():
    assert cli("run", "nonexistent.jsonl", "--task", "text").exit_code == 2
