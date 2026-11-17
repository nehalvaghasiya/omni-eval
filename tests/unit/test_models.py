"""Strict schema and primitive model behaviors."""

import pytest

from omni_eval.types.case import Case
from omni_eval.types.config import Config
from omni_eval.types.prob import Prob
from omni_eval.types.run import Run
from omni_eval.types.score import Score


def test_extra_field_rejected():
    with pytest.raises(ValueError):
        Case.model_validate({"id": "x", "secret_extra": 42})


def test_blank_id_rejected():
    with pytest.raises(ValueError):
        Case(id=" ")


def test_field_paths():
    c = Case(id="x", expected={"label": "A"})
    assert c.get("expected.label") == "A"
    assert c.get("expected.bad") is None


def test_nonfinite_result():
    with pytest.raises(ValueError):
        Score(name="x", value=float("nan"), n_total=1, n_eligible=1, n_scored=1)


def test_invalid_config():
    with pytest.raises(ValueError):
        Config(version=42, task="text", data="file")
    with pytest.raises(ValueError):
        Config(task="text", source="http", data="cases.jsonl")


def test_probability_roundtrip():
    obj = Prob(labels=["a", "b"], values=[0.3, 0.7])
    assert Prob.model_validate_json(obj.model_dump_json()) == obj


def test_run_lookup():
    r = Run(id="abc", task="text", scores=[])
    with pytest.raises(KeyError):
        r.metric("accuracy")


def test_failure_lookup_does_not_flag_perfect_lower_is_better_scores():
    from omni_eval import evaluate

    run = evaluate([Case(id="a", expected="hello", actual="hello")], task="audio")
    assert run.failures() == []


def test_batch_complete_excludes_even_empty_error_strings():
    from omni_eval.types.batch import Batch

    batch = Batch(
        task="text", cases=[Case(id="ok", actual="x"), Case(id="bad", actual="x", error="")]
    )
    assert [c.id for c in batch.complete()] == ["ok"]
    assert [c.id for c in batch.select(frozenset({"actual"}))] == ["ok"]
