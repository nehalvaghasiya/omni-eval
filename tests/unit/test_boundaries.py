"""Numerical edge cases and regressions beyond the all-metric happy paths."""

from math import exp

import pytest

from omni_eval import evaluate
from omni_eval.core.registry import Registry
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.timing import TimingData


def compute(name, cases, settings=None, task="classification"):
    return Registry().get(name).compute(Batch(task=task, cases=cases), settings)


def test_balanced_accuracy_uses_reference_classes():
    data = [
        Case(id=str(i), expected=t, actual=p)
        for i, (t, p) in enumerate([("A", "A"), ("A", "C"), ("B", "B")])
    ]
    assert compute("balanced_accuracy", data).value == pytest.approx(0.75)


@pytest.mark.parametrize("name", ["auroc", "auprc", "brier", "logloss", "topk@1"])
def test_probability_scores_do_not_need_a_predicted_label(name):
    data = [
        Case(id="a", expected="A", probs={"A": 0.8, "B": 0.2}),
        Case(id="b", expected="B", probs={"A": 0.1, "B": 0.9}),
    ]
    result = compute(name, data, {"positive_label": "B"})
    assert result.status == "complete"
    assert result.n_scored == 2


@pytest.mark.parametrize("name", ["auroc", "auprc"])
def test_rank_scores_ties_and_undefined_classes(name):
    tied = [
        Case(id=str(i), expected=t, actual=t, probs={"A": 0.5, "B": 0.5})
        for i, t in enumerate(["A", "B"])
    ]
    assert compute(name, tied, {"positive_label": "B"}).value == 0.5
    undefined = compute(name, tied[:1], {"positive_label": "B"})
    assert undefined.value is None
    assert undefined.details.get("reason")


@pytest.mark.parametrize("name", ["ece", "mce"])
def test_calibration_bad_row_is_partial_not_an_exception(name):
    data = [
        Case(id="a", expected="A", actual="A", probs={"A": 0.8, "B": 0.2}),
        Case(
            id="b",
            expected="B",
            actual="B",
            probs={"labels": ["A", "B"], "values": [0.8, 0.7], "kind": "independent"},
        ),
    ]
    result = compute(name, data)
    assert result.status == "partial"
    assert result.value == pytest.approx(0.2)
    assert result.n_failed == result.n_scored == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_timing_rejects_nonfinite_or_negative(value):
    with pytest.raises(ValueError):
        TimingData(latency_ms=value)


def test_quantiles_have_known_linear_interpolation():
    data = [
        Case(id=str(i), actual="ok", timing=TimingData(latency_ms=v))
        for i, v in enumerate([0, 10, 20, 30])
    ]
    for name, expected in {
        "latency_mean": 15,
        "latency_p50": 15,
        "latency_p95": 28.5,
        "latency_p99": 29.7,
    }.items():
        assert compute(name, data).value == pytest.approx(expected)
    data[0].timing.source = "measured_by_omnieval"
    with pytest.raises(ValueError, match="incompatible"):
        compute("latency_p95", data)


@pytest.mark.parametrize("bad", [["bad"], [float("nan")], [float("inf")], [0.2], [True]])
def test_invalid_logprobs_are_counted_per_case(bad):
    data = [Case(id="good", actual={"logprobs": [-0.5]}), Case(id="bad", actual={"logprobs": bad})]
    result = compute("nll", data, task="text")
    assert result.status == "partial"
    assert result.value == 0.5
    assert result.n_scored == result.n_failed == 1
    assert result.points[-1].status == "error"


def test_token_weighted_loss_and_perplexity_units():
    data = [
        Case(id="a", actual={"logprobs": [-1]}),
        Case(id="b", actual={"logprobs": [-0.25, -0.25, -0.5]}),
    ]
    assert compute("nll", data, task="text").value == 0.5
    result = compute("perplexity", data, task="text")
    assert result.value == pytest.approx(exp(0.5))
    assert result.unit == "ratio"


def test_field_matching_preserves_json_types_and_object_order():
    case = Case(id="typed", expected={"a": 1}, actual={"a": "1"})
    assert compute("fields_f1", [case], task="text").value == 0
    case = Case(id="nested", expected={"a": {"x": 1, "y": 2}}, actual={"a": {"y": 2, "x": 1}})
    assert compute("fields_f1", [case], task="text").value == 1


@pytest.mark.parametrize("task", ["security", "code", "sql"])
def test_init_creates_a_runnable_config_for_every_task(task, tmp_path):
    from typer.testing import CliRunner

    from omni_eval.cli.app import app

    path = tmp_path / task
    result = CliRunner().invoke(app, ["init", str(path), "--task", task])
    assert result.exit_code == 0, result.output
    assert evaluate(path / "eval.yaml").status == "complete"


def test_dry_run_rejects_unknown_task_instead_of_substituting():
    with pytest.raises(ValueError, match="task"):
        evaluate("unused.jsonl", task="typo", dry_run=True)


def test_in_memory_empty_and_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="zero cases"):
        evaluate([], task="text")
    case = Case(id="duplicate", expected="x", actual="x")
    with pytest.raises(ValueError, match="duplicate"):
        evaluate([case, case], task="text")


@pytest.mark.parametrize("name", ["auroc", "auprc", "brier", "logloss", "topk@1", "ece", "mce"])
def test_unknown_probability_truth_is_an_explicit_failed_row(name):
    data = [
        Case(id="a", expected="A", probs={"A": 0.8, "B": 0.2}),
        Case(id="b", expected="C", probs={"A": 0.2, "B": 0.8}),
    ]
    result = compute(name, data, {"positive_label": "B"})
    assert result.n_failed == 1
    assert any(p.id == "b" and p.status == "error" for p in result.points)


def test_single_class_kappa_is_undefined():
    result = compute("kappa", [Case(id="a", expected="A", actual="A")])
    assert result.value is None
    assert result.status == "unsupported"


@pytest.mark.parametrize("name", ["json_valid", "schema_valid"])
def test_nonfinite_python_objects_are_not_json(name):
    result = compute(
        name,
        [Case(id="bad", actual={"x": float("nan")})],
        {"schema": {"type": "object"}},
        task="text",
    )
    assert result.value == 0


def test_empty_error_string_is_still_a_failed_attempt():
    result = compute("accuracy", [Case(id="bad", expected="A", actual="A", error="")])
    assert result.n_failed == 1
    assert result.value is None
