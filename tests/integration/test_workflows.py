"""Real CLI subprocesses and saved artifacts for every bundled task."""

import json
import os
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from omni_eval import evaluate, load
from omni_eval.core.gates import GateEngine
from omni_eval.fixtures.data import names
from omni_eval.io.jsonl import Jsonl
from omni_eval.types.case import Case
from omni_eval.types.config import Config
from omni_eval.types.gate import GateRule

ROOT = Path(__file__).resolve().parents[2]


def cli(*args, cwd=None):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    return subprocess.run(
        [sys.executable, "-m", "omni_eval", *map(str, args)],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )


@pytest.mark.parametrize("task", names())
def test_file_and_cli_examples(task, tmp_path):
    path = ROOT / "examples" / task / "eval.yaml"
    run = evaluate(path, out=tmp_path)
    assert run.status == "complete"
    assert all(s.coverage == 1 for s in run.scores)
    assert load(run.path).summary == run.summary
    real = cli("run", "-c", path, cwd=tmp_path)
    assert real.returncode == 0, real.stderr
    assert "complete" in real.stdout
    demo = cli("demo", task, cwd=tmp_path)
    assert demo.returncode == 0, demo.stderr
    assert "SYNTHETIC" in demo.stdout


def test_cli_saved_workflow_and_gate_exit_codes(tmp_path):
    workspace = tmp_path / "project"
    assert cli("init", workspace).returncode == 0
    run = evaluate(workspace / "eval.yaml", out=tmp_path / "runs")
    assert cli("check", run.path).returncode == 0
    gate = tmp_path / "gate.yaml"
    gate.write_text("gates: [{metric: accuracy, value: 1}]\n")
    assert cli("check", run.path, "-c", gate).returncode == 1
    gate.write_text("gates: [{metric: missing, value: 1}]\n")
    result = cli("check", run.path, "-c", gate, "--strict")
    assert result.returncode == 4, result.stderr
    assert "inconclusive" in result.stdout
    assert cli("run", tmp_path / "absent.jsonl", "--task", "text").returncode == 2
    for fmt in ("html", "json", "md", "junit"):
        rendered = cli("report", run.path, "--format", fmt)
        assert rendered.returncode == 0, rendered.stderr
        if fmt == "json":
            assert json.loads(rendered.stdout)["scores"][0]["value"] == pytest.approx(2 / 3)
        elif fmt == "junit":
            assert ET.fromstring(rendered.stdout).tag == "testsuite"
    assert cli("compare", run.path, run.path).returncode == 0
    replay = Config(
        task="classification", source="replay", data=str(run.path), metrics=["accuracy"]
    )
    assert evaluate(replay).metric("accuracy").value == pytest.approx(2 / 3)


def test_comparison_refuses_different_metric_settings(tmp_path):
    from omni_eval import compare

    path = tmp_path / "cases.jsonl"
    Jsonl().write(path, [{"id": "x", "expected": "hi", "actual": "hi"}])

    def configured(pattern):
        return evaluate(
            Config(
                task="text",
                data=str(path),
                metrics=["format"],
                limits={"metric_settings": {"format": {"pattern": pattern}}},
            )
        )

    with pytest.raises(ValueError, match="settings"):
        compare(configured(".*"), configured("[a-z]+"))


def test_required_pairs_are_not_silently_ignored():
    run = evaluate([Case(id="a", expected="a", actual="a")], task="text")
    result = GateEngine().check(run, [GateRule(metric="exact", value=1, min_pairs=5)])
    assert result[0].status == "inconclusive"


def test_external_network_is_blocked_by_test_fixture():
    with socket.socket() as connection, pytest.raises(PermissionError, match="loopback"):
        connection.connect(("example.invalid", 443))


def test_offline_import_and_demo_do_not_connect(tmp_path):
    code = """
import sys
def audit(event, args):
    if event.startswith('socket.'):
        raise AssertionError('offline command attempted networking: ' + event)
sys.addaudithook(audit)
import omni_eval
from omni_eval.cli.app import app
from typer.testing import CliRunner
result = CliRunner().invoke(app, ['demo', 'rag'])
assert result.exit_code == 0, result.output
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
