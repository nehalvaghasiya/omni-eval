"""Independent hand-calculated classifier and calibration cases."""

import pytest

from omni_eval import evaluate
from omni_eval.core.registry import Registry
from omni_eval.fixtures.data import cases
from omni_eval.types.batch import Batch
from omni_eval.types.prob import Prob


def test_known_f1():
    run = evaluate(
        cases("classification"),
        task="classification",
        metrics=["accuracy", "macro_f1", "micro_f1", "weighted_f1"],
    )
    assert run.metric("accuracy").value == pytest.approx(2 / 3)
    assert run.metric("macro_f1").value == pytest.approx(2 / 3)
    assert run.metric("micro_f1").value == pytest.approx(2 / 3)
    assert run.metric("weighted_f1").value == pytest.approx(2 / 3)
    assert run.metric("macro_f1").points == []


def test_brier_vector():
    data = cases("classification")
    run = evaluate(data, task="classification", metrics=["brier", "ece", "mce", "logloss"])
    assert run.metric("brier").value == pytest.approx((0.02 + 1.28 + 0.32) / 3)
    assert run.metric("ece").value is not None
    assert run.metric("mce").value is not None
    assert run.metric("logloss").value is not None


@pytest.mark.parametrize(
    "metric,expected", [("balanced_accuracy", 0.75), ("kappa", 0.4), ("mcc", 0.5)]
)
def test_corpus(metric, expected):
    run = evaluate(cases("classification"), task="classification", metrics=[metric])
    assert run.metric(metric).value == pytest.approx(expected)


def test_confusion():
    run = evaluate(cases("classification"), task="classification", metrics=["confusion"])
    assert run.metric("confusion").value is None
    assert run.metric("confusion").details == {"labels": ["A", "B"], "matrix": [[1, 1], [0, 1]]}


@pytest.mark.parametrize("name", ["precision", "recall", "auroc", "auprc"])
def test_binary_metrics_require_positive_label(name):
    batch = Batch(task="classification", cases=cases("classification"))
    with pytest.raises(ValueError, match="positive_label"):
        Registry().get(name).compute(batch)


@pytest.mark.parametrize(
    "name",
    [
        "precision",
        "recall",
        "auroc",
        "auprc",
        "macro_precision",
        "macro_recall",
        "micro_precision",
        "weighted_recall",
        "topk@1",
    ],
)
def test_binary_and_averages(name):
    evaluate(cases("classification"), task="classification", metrics=["accuracy"])
    metric = Registry().get(name)
    result = metric.compute(
        Batch(task="classification", cases=cases("classification")), {"positive_label": "B"}
    )
    assert result.value is not None and 0 <= result.value <= 1


@pytest.mark.parametrize(
    "wrong",
    [
        {"labels": ["a", "b"], "values": [0.1, 0.1]},
        {"labels": ["a", "a"], "values": [0.5, 0.5]},
        {"labels": ["a", "b"], "values": [float("nan"), 0.2]},
        {"labels": ["a", "b"], "values": [-0.1, 1.1]},
    ],
)
def test_bad_probs(wrong):
    with pytest.raises(ValueError):
        Prob.model_validate(wrong)


def test_independent_probs_not_softmax():
    good = Prob(labels=["a", "b"], values=[0.7, 0.8], kind="independent")
    assert sum(good.values) > 1
    data = cases("classification")
    data[0] = data[0].model_copy(update={"probs": good})
    run = evaluate(data, task="classification", metrics=["brier"])
    assert run.metric("brier").status == "partial"
    assert run.metric("brier").n_failed == 1


def test_missing_label_and_coverage():
    data = cases("classification")
    data[1] = data[1].model_copy(update={"actual": None})
    result = evaluate(data, task="classification", metrics=["accuracy", "macro_f1"])
    assert result.metric("accuracy").n_scored == 2
    assert result.metric("macro_f1").n_scored == 2
    assert result.metric("accuracy").coverage == pytest.approx(2 / 3)


@pytest.mark.parametrize("invalid", ["topk@0", "topk@1001", "topk@abc", "unknown"])
def test_invalid_metric(invalid):
    with pytest.raises(ValueError):
        Registry().get(invalid)
