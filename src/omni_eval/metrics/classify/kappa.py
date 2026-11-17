"""Cohen kappa."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.metrics.classify.stats import Counts
from omni_eval.types.batch import Batch
from omni_eval.types.score import Score


class Kappa(Metric):
    version = "2"
    name = "kappa"
    tasks = frozenset({"classification", "vision", "audio"})
    needs = frozenset({"label_truth", "label_pred"})
    level = "corpus"

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        selected, points, failed = self.eligible(batch)
        if not selected:
            return self._result(batch, [], points, failed)
        counts = Counts(selected)
        n = sum(map(sum, counts.matrix))
        k = len(counts.labels)
        po = sum(counts.tp(i) for i in range(k)) / n
        pe = sum(counts.support(i) * sum(row[i] for row in counts.matrix) for i in range(k)) / (
            n * n
        )
        if pe >= 1:
            return self._result(
                batch,
                [],
                points,
                failed,
                details={"reason": "kappa is undefined for a single constant class"},
            ).model_copy(update={"status": "unsupported"})
        value = (po - pe) / (1 - pe)
        return self._result(batch, [0.0] * len(selected), points, failed, value=value)
