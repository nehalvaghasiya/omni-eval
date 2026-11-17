"""Replaying stored records never contacts the old prediction target."""

import pytest

from omni_eval import evaluate
from omni_eval.fixtures.data import cases
from omni_eval.types.config import Config


def test_replay_uses_saved_cases_and_new_metric(tmp_path) -> None:
    original = evaluate(cases("text"), task="text", metrics=["exact"], out=tmp_path)
    replayed = evaluate(
        Config(task="text", source="replay", data=str(original.path), metrics=["token_f1"])
    )
    assert replayed.metric("token_f1").value is not None
    assert replayed.metric("token_f1").n_total == 1
    assert replayed.manifest["data_hash"] == original.manifest["data_hash"]
    assert "exact" not in [score.name for score in replayed.scores]
    assert replayed.id != original.id


def test_replay_rejects_run_without_retained_records(tmp_path) -> None:
    from omni_eval.core.engine import Engine

    original = Engine().run(
        cases("text"),
        Config(task="text", data="in-memory", metrics=["exact"], output={"retain": False}),
    )
    from omni_eval.storage.store import Store

    path = Store().save(original, tmp_path)
    assert not (path / "cases.jsonl").exists()
    assert not (path / "points.jsonl").exists()
    assert not (path / "errors.jsonl").exists()
    with pytest.raises(ValueError, match="did not retain"):
        evaluate(Config(task="text", source="replay", data=str(path), metrics=["exact"]))


def test_replay_rejects_tampered_source(tmp_path) -> None:
    original = evaluate(cases("text"), task="text", out=tmp_path)
    assert original.path is not None
    (original.path / "cases.jsonl").write_text('{"id":"forged"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        evaluate(Config(task="text", source="replay", data=str(original.path)))
