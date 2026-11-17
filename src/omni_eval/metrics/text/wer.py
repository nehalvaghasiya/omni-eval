"""Words edit error rate with documented empty reference behavior."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.metrics.data import answer
from omni_eval.metrics.text.distance import distance
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class Wer(Metric):
    name = "wer"
    needs = frozenset({"expected", "actual"})
    tasks = frozenset({"audio", "text", "rag", "vision"})
    direction = "down"

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        a = answer(case)
        b = answer(case, truth=True)
        if a is None or b is None:
            raise ValueError("transcript text required")
        predicted = a.split()
        reference = b.split()
        if not reference:
            if predicted:
                raise ValueError("zero-length reference cannot define error rate")
            return 0.0
        return distance(predicted, reference) / len(reference)

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
