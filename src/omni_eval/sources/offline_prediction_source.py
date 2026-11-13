from collections.abc import Iterable
from pathlib import Path

from omni_eval.schemas.eval_sample import EvalSample
from omni_eval.schemas.prediction_record import PredictionRecord
from omni_eval.sources.jsonl_loader import load_jsonl
from omni_eval.sources.parquet_loader import load_parquet


class OfflinePredictionSource:
    """Read predictions without model inference."""

    def __init__(self, path: str):
        self.path = path

    def acquire(self, samples: Iterable[EvalSample]) -> list[PredictionRecord]:
        if Path(self.path).suffix.lower() == ".parquet":
            return load_parquet(self.path, PredictionRecord)

        return load_jsonl(self.path, PredictionRecord)
