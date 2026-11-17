"""multiclass Matthews correlation coefficient."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.metrics.classify.stats import Counts
from omni_eval.types.batch import Batch
from omni_eval.types.score import Score


class Mcc(Metric):
    name = "mcc"
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
        diag = sum(counts.tp(i) for i in range(k))
        t = [counts.support(i) for i in range(k)]
        p = [sum(row[i] for row in counts.matrix) for i in range(k)]
        value = (
            n * diag - sum(row[i] for i in range(k) for row in [])
            if False
            else (n * diag - sum(p[i] * t[i] for i in range(k)))
            / ((n * n - sum((x * x for x in p))) * (n * n - sum(x * x for x in t))) ** 0.5
            if (n * n - sum(x * x for x in p)) * (n * n - sum(x * x for x in t)) > 0
            else 0.0
        )
        return self._result(batch, [0.0] * len(selected), points, failed, value=value)
