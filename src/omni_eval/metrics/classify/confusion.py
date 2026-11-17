"""Structured corpus confusion matrix; not a fabricated scalar."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.metrics.classify.stats import Counts
from omni_eval.types.batch import Batch
from omni_eval.types.score import Score


class Confusion(Metric):
    name = "confusion"
    tasks = frozenset({"classification", "vision", "audio"})
    needs = frozenset({"label_truth", "label_pred"})
    level = "corpus"
    direction = "none"
    unit = "report"

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        selected, points, failed = self.eligible(batch)
        counts = Counts(selected) if selected else None
        return self._result(
            batch,
            [0.0] * len(selected),
            points,
            failed,
            details={"labels": counts.labels, "matrix": counts.matrix} if counts else {},
        ).model_copy(update={"value": None})
