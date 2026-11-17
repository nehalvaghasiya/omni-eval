"""Old schema-1.0 run folders stay readable without a migration step."""

import json

from omni_eval import load
from omni_eval.io.hash import hash_file
from omni_eval.io.jsonl import Jsonl


def test_saved_legacy_format(tmp_path):
    folder = tmp_path / "old"
    folder.mkdir()
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_id": "oldrun",
                "status": "completed",
                "source_mode": "offline",
                "metadata": {"task": "classification"},
                "is_fixture": True,
            }
        )
    )
    (folder / "metrics.json").write_text(
        json.dumps(
            [
                {
                    "name": "accuracy",
                    "value": 1.0,
                    "status": "completed",
                    "n_total": 1,
                    "n_scored": 1,
                    "n_eligible": 1,
                    "coverage": 1.0,
                    "details": {"metric_version": "1"},
                }
            ]
        )
    )
    Jsonl().write(
        folder / "samples.jsonl",
        [
            {
                "sample_id": "1",
                "input": {"text": "hello"},
                "reference": {"label": "greeting"},
                "metadata": {},
            }
        ],
    )
    Jsonl().write(
        folder / "predictions.jsonl",
        [
            {
                "sample_id": "1",
                "output": {"label": "greeting"},
            }
        ],
    )
    Jsonl().write(
        folder / "sample_scores.jsonl",
        [
            {
                "sample_id": "1",
                "metric": "accuracy",
                "status": "completed",
                "value": 1.0,
            }
        ],
    )
    Jsonl().write(folder / "errors.jsonl", [])
    (folder / "checksums.json").write_text(
        json.dumps({path.name: hash_file(path) for path in folder.iterdir() if path.is_file()})
    )
    run = load(folder)
    assert run.id == "oldrun"
    assert run.metric("accuracy").value == 1
    assert run.cases[0].actual == {"label": "greeting"}
    assert len(run.metric("accuracy").points) == 1
