"""Hand-computed exact, token, JSON, edit-distance and syntax checks."""

import pytest

from omni_eval import evaluate
from omni_eval.core.registry import Registry
from omni_eval.fixtures.data import cases
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case


def test_exact_and_token():
    run = evaluate(
        cases("text"),
        task="text",
        metrics=["exact", "answer_match", "token_f1", "token_precision", "token_recall"],
    )
    assert run.metric("exact").value == 0
    for name in ["answer_match", "token_f1", "token_precision", "token_recall"]:
        assert run.metric(name).value == 1


@pytest.mark.parametrize("text,expected", [("{}", 1), ("{broken", 0), ("null", 1), ("NaN", 0)])
def test_json(text, expected):
    run = evaluate([Case(id="j", actual=text)], task="text", metrics=["json_valid"])
    assert run.metric("json_valid").value == expected


def test_schema_and_fields():
    data = [Case(id="x", actual={"a": 1, "b": 2}, expected={"a": 1, "b": 3})]
    run = evaluate(data, task="text", metrics=["fields_f1"])
    assert run.metric("fields_f1").value == pytest.approx(0.5)
    schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}}
    metric = Registry().get("schema_valid")
    assert metric.compute(Batch(task="text", cases=data), {"schema": schema}).value == 1


@pytest.mark.parametrize(
    "task,name,expected",
    [
        ("audio", "wer", 0.5),
        ("audio", "cer", 1 / 11),
        ("code", "code_valid", 1),
        ("sql", "sql_select_safe", 1),
    ],
)
def test_other(task, name, expected):
    run = evaluate(cases(task), task=task, metrics=[name])
    assert run.metric(name).value == pytest.approx(expected)


def test_code_never_executes(tmp_path):
    marker = tmp_path / "should-not-exist"
    code = f"__import__('pathlib').Path({str(marker)!r}).write_text('bad')"
    run = evaluate([Case(id="x", actual=code)], task="code", metrics=["code_valid"])
    assert run.metric("code_valid").value == 1
    assert not marker.exists()


def test_no_reference_rejected():
    with pytest.raises(ValueError, match="required references"):
        evaluate([Case(id="x", actual="hello")], task="text", metrics=["exact"])


def test_undefined_edits_are_not_zeros():
    data = [Case(id="x", actual="hi", expected="")]
    run = evaluate(data, task="audio", metrics=["wer"])
    assert run.metric("wer").value is None
    assert run.metric("wer").n_failed == 1
