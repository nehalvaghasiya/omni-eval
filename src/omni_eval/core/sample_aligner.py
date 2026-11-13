from collections.abc import Iterable

from omni_eval.exceptions import InputError
from omni_eval.schemas.eval_sample import EvalSample
from omni_eval.schemas.prediction_record import PredictionRecord


class SampleAligner:
    """Join by explicit IDs; never by row order."""

    def align(
        self, samples: Iterable[EvalSample], predictions: Iterable[PredictionRecord]
    ) -> tuple[list[tuple[EvalSample, PredictionRecord | None]], list[str]]:
        unique_samples: dict[str, EvalSample] = {}

        for sample in samples:
            if sample.sample_id in unique_samples:
                raise InputError(f"duplicate sample: {sample.sample_id}")

            unique_samples[sample.sample_id] = sample

        if not unique_samples:
            raise InputError("empty sample dataset")

        unique_predictions: dict[str, PredictionRecord] = {}

        for prediction in predictions:
            if prediction.sample_id in unique_predictions:
                raise InputError(f"duplicate prediction: {prediction.sample_id}")

            unique_predictions[prediction.sample_id] = prediction

        extras = sorted(set(unique_predictions) - set(unique_samples))

        return [
            (sample, unique_predictions.get(sample_id))
            for sample_id, sample in unique_samples.items()
        ], extras
