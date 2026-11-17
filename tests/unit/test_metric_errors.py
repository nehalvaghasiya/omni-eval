"""Meaningful missing-input and invalid-configuration paths across metric families."""

import pytest

from omni_eval.core.registry import Registry
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case


@pytest.mark.parametrize(
    "task,name,case,settings",
    [
        ("code", "code_valid", {"actual": {}}, {}),
        ("code", "code_valid", {"actual": "x = 1"}, {"language": "javascript"}),
        ("sql", "sql_select_safe", {"actual": {}}, {}),
        ("text", "exact", {"actual": {}, "expected": {}}, {}),
        ("text", "token_f1", {"actual": {}, "expected": {}}, {}),
        ("text", "fields_f1", {"actual": "x", "expected": "x"}, {}),
        ("text", "format", {"actual": {}}, {}),
        ("text", "format", {"actual": "x"}, {"pattern": "x" * 101}),
        ("text", "schema_valid", {"actual": {}}, {}),
        ("text", "schema_valid", {"actual": {}}, {"schema": {"type": "not-a-type"}}),
        (
            "text",
            "schema_valid",
            {"actual": {}},
            {"schema": {"$ref": "https://example.invalid/schema"}},
        ),
        ("audio", "wer", {"actual": {}, "expected": {}}, {}),
        ("audio", "cer", {"actual": {}, "expected": {}}, {}),
        ("rag", "citation_valid", {"actual": {}, "contexts": [{"id": "a", "rank": 1}]}, {}),
        ("agent", "tool_allowed", {"expected": {}, "trace": {"steps": []}}, {}),
        ("agent", "step_order", {"expected": {}, "trace": {"steps": []}}, {}),
        ("agent", "state_partial", {"expected": {"state": {}}, "trace": {"steps": []}}, {}),
        ("agent", "tool_success", {"trace": {"steps": []}}, {}),
        ("agent", "tool_success", {"trace": {"steps": [{"name": "x", "kind": "tool_call"}]}}, {}),
        ("security", "attack_success", {"expected": "x", "actual": "x"}, {}),
        ("security", "attack_success", {"expected": {}, "actual": {}}, {}),
        ("security", "policy_violation", {"actual": "x"}, {}),
        ("security", "policy_violation", {"actual": {}}, {}),
    ],
)
def test_bad_inputs_are_explicit_errors(task, name, case, settings):
    result = (
        Registry().get(name).compute(Batch(task=task, cases=[Case(id="bad", **case)]), settings)
    )
    assert result.value is None
    assert result.n_failed == 1
    assert result.n_scored == 0
    assert result.status == "error"
    assert result.points[0].reason


@pytest.mark.parametrize(
    "task,name,value,settings,expected",
    [
        ("code", "code_valid", "def broken(:", {}, 0),
        ("sql", "sql_select_safe", "DELETE FROM users", {}, 0),
        ("sql", "sql_select_safe", "SELECT 1; DROP TABLE users", {}, 0),
        ("text", "format", "long", {"max_length": 2}, 0),
        ("text", "format", "x", {"min_length": 2}, 0),
        ("text", "format", "abc", {"pattern": "[0-9]+"}, 0),
        ("text", "schema_valid", '{"a": 1}', {"schema": {"type": "object"}}, 1),
        ("text", "schema_valid", "broken", {"schema": {"type": "object"}}, 0),
        ("text", "schema_valid", "NaN", {"schema": {"type": "number"}}, 0),
    ],
)
def test_valid_scoring_failures_are_numeric_zero(task, name, value, settings, expected):
    result = (
        Registry().get(name).compute(Batch(task=task, cases=[Case(id="x", actual=value)]), settings)
    )
    assert result.value == expected
    assert result.status == "complete"
    assert result.n_failed == 0


@pytest.mark.parametrize(
    "name",
    [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "mcc",
        "kappa",
        "precision",
        "recall",
        "latency_mean",
        "nll",
    ],
)
def test_empty_batches_have_no_fabricated_zero(name):
    metric = Registry().get(name)
    result = metric.compute(Batch(task=next(iter(metric.tasks)), cases=[]))
    assert result.value is None
    assert result.coverage == 0


@pytest.mark.parametrize("name", ["ece", "mce"])
@pytest.mark.parametrize("bins", [1, 101])
def test_invalid_calibration_bins(name, bins):
    with pytest.raises(ValueError, match="bins"):
        Registry().get(name).compute(Batch(task="classification", cases=[]), {"bins": bins})


def test_empty_text_and_likelihood_prerequisites():
    batch = Batch(task="text", cases=[Case(id="a", expected="", actual="")])
    assert Registry().get("token_f1").compute(batch).value == 1
    assert Registry().get("wer").compute(batch).value == 0
    assert Registry().get("cer").compute(batch).value == 0
    for actual in ({}, {"logprobs": []}):
        result = (
            Registry().get("nll").compute(Batch(task="text", cases=[Case(id="b", actual=actual)]))
        )
        assert result.value is None
        assert result.n_missing == 1
        assert result.points[0].status == "missing_input"
    overflow = (
        Registry()
        .get("perplexity")
        .compute(Batch(task="text", cases=[Case(id="b", actual={"logprobs": [-1000]})]))
    )
    assert overflow.value is None
    assert overflow.status == "unsupported"


def test_local_json_schema_references_work_without_fetching():
    schema = {"$defs": {"answer": {"type": "integer"}}, "$ref": "#/$defs/answer"}
    batch = Batch(task="text", cases=[Case(id="a", actual=1), Case(id="b", actual="oops")])
    result = Registry().get("schema_valid").compute(batch, {"schema": schema})
    assert result.value == 0.5
    assert result.n_failed == 0
