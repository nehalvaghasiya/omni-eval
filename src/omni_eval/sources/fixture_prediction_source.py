from collections.abc import Iterable

from omni_eval.fixtures.fixture_predictions import classification_fixture
from omni_eval.schemas.eval_sample import EvalSample
from omni_eval.schemas.prediction_record import PredictionRecord


class FixturePredictionSource:
    """Deterministic local source, never a claim of measured quality."""

    def acquire(self, samples: Iterable[EvalSample]) -> list[PredictionRecord]:
        return classification_fixture()[1]
