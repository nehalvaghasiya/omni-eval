"""Independent reference-library checks; optional stats profile only."""

import pytest

from omni_eval.core.registry import Registry
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case


def test_classifier_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics", reason="optional stats extra is absent")
    truth = ["A", "A", "B", "B", "C", "C"]
    pred = ["A", "B", "B", "C", "C", "A"]
    batch = Batch(
        task="classification",
        cases=[
            Case(id=str(i), expected=t, actual=p)
            for i, (t, p) in enumerate(zip(truth, pred, strict=True))
        ],
    )
    refs = {
        "accuracy": sk.accuracy_score(truth, pred),
        "balanced_accuracy": sk.balanced_accuracy_score(truth, pred),
        "mcc": sk.matthews_corrcoef(truth, pred),
        "kappa": sk.cohen_kappa_score(truth, pred),
    }
    for average in ("macro", "micro", "weighted"):
        for label, func in (
            ("precision", sk.precision_score),
            ("recall", sk.recall_score),
            ("f1", sk.f1_score),
        ):
            refs[f"{average}_{label}"] = func(truth, pred, average=average, zero_division=0)
    for name, value in refs.items():
        assert Registry().get(name).compute(batch).value == pytest.approx(value), name


@pytest.mark.parametrize("values", [[0.1, 0.8, 0.6, 0.4], [0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]])
def test_probability_metrics_match_sklearn(values):
    sk = pytest.importorskip("sklearn.metrics", reason="optional stats extra is absent")
    truth = [0, 1, 0, 1]
    batch = Batch(
        task="classification",
        cases=[
            Case(id=str(i), expected=str(t), probs={"0": 1 - p, "1": p})
            for i, (t, p) in enumerate(zip(truth, values, strict=True))
        ],
    )
    settings = {"positive_label": "1"}
    assert Registry().get("auroc").compute(batch, settings).value == pytest.approx(
        sk.roc_auc_score(truth, values)
    )
    assert Registry().get("auprc").compute(batch, settings).value == pytest.approx(
        sk.average_precision_score(truth, values)
    )
    # OmniEval uses the whole class vector: twice the binary positive-class Brier.
    assert Registry().get("brier").compute(batch).value == pytest.approx(
        2 * sk.brier_score_loss(truth, values)
    )
