"""Real file ingest, legacy split alignment, integrity and retention."""

import pytest

from omni_eval import evaluate, load
from omni_eval.core.align import Aligner
from omni_eval.io.jsonl import Jsonl
from omni_eval.types.case import Case
from omni_eval.types.config import Config


def test_inline_files(tmp_path):
    file = tmp_path / "cases.jsonl"
    Jsonl().write(
        file,
        [
            {"id": "1", "expected": "ok", "actual": "ok"},
            {"id": "2", "expected": "yes", "actual": "no"},
        ],
    )
    run = evaluate(file, task="text", metrics=["exact"], out=tmp_path / "runs")
    assert run.metric("exact").value == 0.5
    assert load(run.path).metric("exact").value == 0.5


def test_legacy_split_alignment(tmp_path):
    samples = tmp_path / "samples.jsonl"
    pred = tmp_path / "preds.jsonl"
    Jsonl().write(
        samples,
        [
            {"sample_id": "a", "reference": {"label": "A"}},
            {"sample_id": "b", "reference": {"label": "B"}},
        ],
    )
    Jsonl().write(
        pred,
        [
            {"sample_id": "b", "output": {"label": "B"}},
            {"sample_id": "a", "output": {"label": "A"}},
        ],
    )
    run = evaluate(
        Config(
            task="classification", data=str(samples), predictions=str(pred), metrics=["accuracy"]
        )
    )
    assert run.metric("accuracy").value == 1


def test_missing_prediction_is_visible(tmp_path):
    samples = tmp_path / "samples.jsonl"
    pred = tmp_path / "preds.jsonl"
    Jsonl().write(samples, [{"id": "a", "expected": "hi"}, {"id": "b", "expected": "world"}])
    Jsonl().write(pred, [{"id": "a", "actual": "hi"}])
    run = evaluate(Config(task="text", data=str(samples), predictions=str(pred), metrics=["exact"]))
    assert run.metric("exact").value == 1
    assert run.metric("exact").coverage == 0.5
    assert run.metric("exact").status == "partial"


def test_duplicate_predictions():
    with pytest.raises(ValueError, match="duplicate"):
        Aligner().join([Case(id="a")], [{"id": "a", "actual": "x"}, {"id": "a", "actual": "y"}])


def test_extra_prediction():
    with pytest.raises(ValueError, match="unknown"):
        Aligner().join([Case(id="a")], [{"id": "other", "actual": "x"}])


def test_jsonl_invalid_line(tmp_path):
    f = tmp_path / "bad.jsonl"
    f.write_text("{invalid}\n")
    with pytest.raises(ValueError, match="invalid JSON"):
        list(Jsonl().read(f))


def test_tamper_detected(tmp_path):
    run = evaluate([Case(id="a", expected="yes", actual="yes")], task="text", out=tmp_path)
    (run.path / "scores.json").write_text("[]")
    with pytest.raises(ValueError, match="integrity"):
        load(run.path)


def test_retention_false(tmp_path):
    c = Case(id="secret", input="PRIVATE", expected="ok", actual="ok")
    conf = Config(task="text", data="memory", metrics=["exact"], output={"retain": False})
    evaluate(conf, dry_run=True, out=None)
    from omni_eval.core.engine import Engine

    actual = Engine().run([c], conf)
    from omni_eval.storage.store import Store

    folder = Store().save(actual, tmp_path)
    assert not (folder / "cases.jsonl").exists()
    assert not (folder / "points.jsonl").exists()
    assert "PRIVATE" not in " ".join(x.read_text() for x in folder.glob("*.json"))


def test_reject_run_collision(tmp_path):
    from omni_eval.storage.store import Store

    run = evaluate([Case(id="x", actual="ok", expected="ok")], task="text")
    Store().save(run, tmp_path)
    with pytest.raises(FileExistsError):
        Store().save(run, tmp_path)
