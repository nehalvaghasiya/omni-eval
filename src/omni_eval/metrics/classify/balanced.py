"""mean recall across all classes."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.metrics.classify.stats import Counts
from omni_eval.types.batch import Batch
from omni_eval.types.score import Score


class Balanced(Metric):
    version = "2"
    name = "balanced_accuracy"
    tasks = frozenset({"classification", "vision", "audio"})
    needs = frozenset({"label_truth", "label_pred"})
    level = "corpus"

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        selected, points, failed = self.eligible(batch)
        if not selected:
            return self._result(batch, [], points, failed)
        counts = Counts(selected)
        represented = [i for i in range(len(counts.labels)) if counts.support(i)]
        value = sum(counts.rate(i, "recall") for i in represented) / len(represented)
        return self._result(batch, [0.0] * len(selected), points, failed, value=value)
