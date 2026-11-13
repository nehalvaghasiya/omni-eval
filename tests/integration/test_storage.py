import json
from pathlib import Path

import pytest

from omni_eval.api import evaluate, load_run
from omni_eval.exceptions import InputError
from omni_eval.schemas.prediction_record import PredictionRecord
from omni_eval.sources.jsonl_loader import load_jsonl, write_jsonl
from omni_eval.storage.file_run_repository import FileRunRepository


def test_run_persist_and_hash_integrity(tmp_path, classifier):
    samples, predictions = classifier
    run = evaluate(samples, predictions, ["accuracy"], output_dir=tmp_path)

    assert set((Path(run.path)).iterdir()) >= {
        Path(run.path) / name
        for name in (
            "manifest.json",
            "metrics.json",
            "samples.jsonl",
            "predictions.jsonl",
            "sample_scores.jsonl",
            "errors.jsonl",
            "checksums.json",
            "report.html",
        )
    }
    assert load_run(run.path).metrics[0].value == run.metrics[0].value
    (Path(run.path) / "metrics.json").write_text("[]")

    with pytest.raises(InputError, match="integrity"):
        load_run(run.path)


def test_duplicate_and_malformed_jsonl(tmp_path):
    file = tmp_path / "pred.jsonl"
    item = '{"sample_id":"a","output":{}}\n'
    file.write_text(item + item)

    with pytest.raises(InputError, match="duplicate"):
        load_jsonl(file, PredictionRecord)

    file.write_text('{"broken":\n')

    with pytest.raises(InputError):
        load_jsonl(file, PredictionRecord)


def test_empty_jsonl_refused(tmp_path):
    file = tmp_path / "empty.jsonl"
    file.touch()

    with pytest.raises(InputError, match="empty"):
        load_jsonl(file, PredictionRecord)


def test_csv_explicit_mapping_embedded_newline(tmp_path):
    import csv

    from omni_eval.schemas.eval_sample import EvalSample
    from omni_eval.sources.csv_loader import load_csv

    file = tmp_path / "cases.csv"

    with file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "input_col", "reference_col"])
        writer.writerow(
            [
                "id-1",
                json.dumps({"text": "hello,\nworld"}),
                json.dumps({"text": "hello"}),
            ]
        )

    rows = load_csv(
        file,
        EvalSample,
        {"sample_id": "id", "input": "input_col", "reference": "reference_col"},
    )

    assert rows[0].input["text"] == "hello,\nworld"


def test_atomic_run_rejects_collision(tmp_path, classifier):
    samples, predictions = classifier
    run = evaluate(samples, predictions, ["accuracy"])
    FileRunRepository().save(run, tmp_path)

    with pytest.raises(InputError, match="collision"):
        FileRunRepository().save(run, tmp_path)


def test_missing_prediction_errors_are_reloadable(tmp_path, classifier):
    samples, predictions = classifier
    run = evaluate(samples, predictions[:1], ["accuracy"], output_dir=tmp_path)

    assert len(load_run(run.path).errors) == 2
