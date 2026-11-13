import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from omni_eval.cli.app import app

PACKAGE_PATH = str(Path(__file__).resolve().parents[2] / "src")


@pytest.mark.parametrize(
    "arguments",
    [
        "init --dry-run",
        "validate --dry-run",
        "evaluate --dry-run",
        "compare --dry-run",
        "replay --dry-run",
        "report --dry-run",
        "doctor --dry-run",
        "metrics list --dry-run",
        "metrics inspect --dry-run",
        "judge --dry-run",
        "runner --dry-run",
        "serve --dry-run",
    ],
)
def test_airgapped_cli_commands(arguments, tmp_path):
    result = CliRunner().invoke(app, arguments.split())

    assert result.exit_code == 0, result.output
    assert "DRY RUN" in result.output


def test_bad_input_returns_exit_two(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omni_eval",
            "evaluate",
            "--samples",
            "missing.jsonl",
            "--predictions",
            "missingpred.jsonl",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        env={**os.environ, "PYTHONPATH": PACKAGE_PATH},
    )

    assert result.returncode == 2


def test_help_and_version():
    for arguments in [("--help",), ("--version",)]:
        result = subprocess.run(
            [sys.executable, "-m", "omni_eval", *arguments],
            capture_output=True,
            text=True,
            timeout=20,
            env={**os.environ, "PYTHONPATH": PACKAGE_PATH},
        )

        assert result.returncode == 0, result.stderr
