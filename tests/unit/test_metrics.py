import math

import pytest

from omni_eval.api import evaluate
from omni_eval.core.metric_registry import MetricRegistry
from omni_eval.exceptions import InputError
from omni_eval.metrics.classification.brier_score import brier_score
from omni_eval.metrics.classification.expected_calibration_error import (
    calibration_error,
)
from omni_eval.metrics.retrieval.ranking import ranking_metric
from omni_eval.metrics.text.token_f1 import token_scores
from omni_eval.metrics.text.word_error_rate import error_rate
from omni_eval.schemas.eval_sample import EvalSample
from omni_eval.schemas.prediction_record import PredictionRecord


def test_hand_calculated_classification(classifier):
    samples, predictions = classifier
    run = evaluate(
        samples,
        predictions,
        ["accuracy", "macro_f1", "brier_score", "ece", "confusion_matrix"],
    )
    data = {metric.name: metric for metric in run.metrics}

    assert data["accuracy"].value == pytest.approx(2 / 3)
    assert data["macro_f1"].value == pytest.approx(2 / 3)
    assert data["brier_score"].value == pytest.approx((0.02 + 0.08 + 0.72) / 3)
    assert data["ece"].value == pytest.approx(0.3)
    assert data["confusion_matrix"].value is None
    assert data["confusion_matrix"].details["labels"] == ["no", "yes"]
    assert all(
        score.value is None for score in run.sample_scores if score.metric == "macro_f1"
    )


def test_hand_calculated_probability():
    assert brier_score(
        ["y", "n"], ["n", "y"], [[0.2, 0.8], [0.7, 0.3]], positive_class="y"
    ) == pytest.approx(0.065)
    assert calibration_error(
        ["yes"], ["no", "yes"], [[0.1, 0.9]], bins=5
    ) == pytest.approx(0.1)


@pytest.mark.parametrize(
    "metric, expected",
    [
        ("precision_at_k", 0.5),
        ("recall_at_k", 0.5),
        ("hit_at_k", 1),
        ("mrr", 0.5),
        ("map", 0.25),
        ("ndcg_at_k", 1 / math.log2(3) / (1 + 1 / math.log2(3))),
        ("r_precision", 0.5),
    ],
)
def test_rank_metrics(metric, expected):
    assert ranking_metric(
        metric, ["junk", "good", "other"], {"good", "absent"}, k=2
    ) == pytest.approx(expected)


def test_ranking_requires_labels():
    with pytest.raises(ValueError):
        ranking_metric("recall_at_k", ["a"], set())


def test_token_f1():
    assert token_scores("A B", "a c") == pytest.approx((0.5, 0.5, 0.5))
    assert token_scores("", "") == (1, 1, 1)


def test_asr_error_rates():
    assert error_rate("hello there", "hello world") == pytest.approx(0.5)
    assert error_rate("cat", "cut", characters=True) == pytest.approx(1 / 3)


def test_text_metrics():
    sample = EvalSample(
        sample_id="t1", input={"query": "hi"}, reference={"answer": "Hello, world!"}
    )
    pred = PredictionRecord(sample_id="t1", output={"answer": "hello world"})
    run = evaluate(
        [sample], [pred], ["exact_match", "normalized_exact_match", "token_f1"]
    )

    assert [m.value for m in run.metrics] == pytest.approx([0, 1, 1])


def test_structured_json():
    sample = EvalSample(
        sample_id="j",
        input={
            "schema": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            }
        },
    )
    pred = PredictionRecord(sample_id="j", output={"text": '{"x": 1}'})
    run = evaluate([sample], [pred], ["json_validity", "json_schema_validity"])

    assert all(metric.value == 1 for metric in run.metrics)


def test_missing_reference_not_fabricated(classifier):
    samples, predictions = classifier

    for sample in samples:
        sample.reference = None

    with pytest.raises(InputError, match="reference.label"):
        evaluate(samples, predictions, ["accuracy"])


def test_missing_probabilities_not_zero(classifier):
    samples, predictions = classifier

    for p in predictions:
        p.probabilities = None

    run = evaluate(samples, predictions, ["brier_score"])

    assert run.metrics[0].status == "insufficient_input"
    assert run.metrics[0].value is None


def test_missing_prediction_counts_in_denominator(classifier):
    samples, predictions = classifier
    run = evaluate(samples, predictions[:-1], ["accuracy"])

    assert run.metrics[0].n_total == 3
    assert run.metrics[0].n_missing == 1
    assert run.metrics[0].coverage == pytest.approx(2 / 3)
    assert run.manifest.status == "partial"


def test_extra_predictions_recoverable(classifier):
    samples, predictions = classifier
    predictions.append(PredictionRecord(sample_id="stranger", output={"label": "yes"}))

    assert "extra prediction" in evaluate(samples, predictions, ["accuracy"]).errors[0]


def test_unknown_metric_rejected(classifier):
    samples, predictions = classifier

    with pytest.raises(InputError, match="unknown"):
        evaluate(samples, predictions, ["superficial_magic_quality"])


def test_registered_catalog_consistent():
    assert "accuracy" in [m.id for m in MetricRegistry().list()]
