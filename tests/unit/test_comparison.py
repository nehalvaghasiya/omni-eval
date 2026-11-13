from typing import get_type_hints

import pytest

from omni_eval.api import compare, evaluate
from omni_eval.comparison.gate_evaluator import GateEvaluator
from omni_eval.comparison.slice_comparator import SliceComparator
from omni_eval.exceptions import InputError
from omni_eval.schemas.comparison_result import ComparisonResult
from omni_eval.schemas.prediction_record import PredictionRecord


def test_paired_baseline_vs_candidate(classifier):
    samples, predictions = classifier
    improved = [PredictionRecord.model_validate(p.model_dump()) for p in predictions]
    improved[2].output["label"] = "yes"
    left = evaluate(samples, predictions, ["accuracy", "macro_f1"])
    right = evaluate(
        list(reversed(samples)), list(reversed(improved)), ["accuracy", "macro_f1"]
    )
    result = compare(left, right, iterations=100, seed=10)

    assert {r.metric for r in result} == {"accuracy", "macro_f1"}
    assert next(r for r in result if r.metric == "accuracy").delta == pytest.approx(
        1 / 3
    )
    assert result[0].interval is not None
    assert (
        compare(left, right, iterations=100, seed=10)[0].interval == result[0].interval
    )


def test_comparison_refuses_different_dataset(classifier):
    samples, predictions = classifier
    left = evaluate(samples, predictions, ["accuracy"])
    samples[0].input["text"] = "mutated"
    right = evaluate(samples, predictions, ["accuracy"])

    with pytest.raises(InputError, match="dataset"):
        compare(left, right, iterations=100)


def test_compare_public_return_annotation_resolves():
    assert get_type_hints(compare)["return"] == list[ComparisonResult]


def test_bootstrap_callbacks_keep_each_metrics_inputs(classifier, monkeypatch):
    samples, predictions = classifier
    improved = [PredictionRecord.model_validate(p.model_dump()) for p in predictions]
    improved[2].output["label"] = "yes"
    left = evaluate(samples, predictions, ["accuracy", "brier_score"])
    right = evaluate(samples, improved, ["accuracy", "brier_score"])
    callbacks = []

    def capture_callback(n, callback, **settings):
        callbacks.append((n, callback))

        return (0.0, 0.0)

    monkeypatch.setattr(
        "omni_eval.comparison.paired_comparator.paired_bootstrap_statistic",
        capture_callback,
    )
    results = compare(left, right, iterations=100)

    assert [result.metric for result in results] == ["accuracy", "brier_score"]
    assert [callback(list(range(n))) for n, callback in callbacks] == pytest.approx(
        [1 / 3, 0.0]
    )


def test_gate_pass_fail_inconclusive(classifier):
    samples, predictions = classifier
    full = evaluate(samples, predictions, ["accuracy"]).metrics[0]
    gate = GateEvaluator()

    assert gate.evaluate(full, min_value=0.5).status == "pass"
    assert gate.evaluate(full, min_value=0.99).status == "fail"
    partial = evaluate(samples, predictions[:-1], ["accuracy"]).metrics[0]

    assert gate.evaluate(partial, min_value=0.5).status == "inconclusive"


def test_slice_min_support(classifier):
    samples, predictions = classifier

    for sample in samples:
        sample.metadata["slice"] = "demo"

    run = evaluate(samples, predictions, ["accuracy"])

    assert SliceComparator().summarize(run, "accuracy", min_support=5) == {}
    assert SliceComparator().summarize(run, "accuracy", min_support=2)["demo"]["n"] == 3
