from collections.abc import Iterable

from omni_eval.schemas.eval_sample import EvalSample
from omni_eval.schemas.prediction_record import PredictionRecord
from omni_eval.storage.file_run_repository import FileRunRepository


class ReplayPredictionSource:
    """Use recorded canonical predictions; no inference or side effects."""

    def __init__(self, run_path: str):
        self.run_path = run_path

    def acquire(self, samples: Iterable[EvalSample]) -> list[PredictionRecord]:
        return FileRunRepository().load(self.run_path).predictions
