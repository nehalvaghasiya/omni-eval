"""Hand-auditable synthetic records; never real benchmark outputs."""

from omni_eval.schemas.eval_sample import EvalSample
from omni_eval.schemas.prediction_record import PredictionRecord


def classification_fixture() -> tuple[list[EvalSample], list[PredictionRecord]]:
    samples = [
        EvalSample(
            sample_id="fixture-1", input={"text": "A"}, reference={"label": "yes"}
        ),
        EvalSample(
            sample_id="fixture-2", input={"text": "B"}, reference={"label": "no"}
        ),
        EvalSample(
            sample_id="fixture-3", input={"text": "C"}, reference={"label": "yes"}
        ),
    ]
    predictions = [
        PredictionRecord(
            sample_id="fixture-1",
            output={"label": "yes"},
            probabilities={
                "labels": ["no", "yes"],
                "values": [0.1, 0.9],
                "kind": "softmax",
            },
        ),
        PredictionRecord(
            sample_id="fixture-2",
            output={"label": "no"},
            probabilities={
                "labels": ["no", "yes"],
                "values": [0.8, 0.2],
                "kind": "softmax",
            },
        ),
        PredictionRecord(
            sample_id="fixture-3",
            output={"label": "no"},
            probabilities={
                "labels": ["no", "yes"],
                "values": [0.6, 0.4],
                "kind": "softmax",
            },
        ),
    ]

    return samples, predictions
