"""Corpus F1 macro/micro/weighted; never mean per-row F1."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.metrics.classify.stats import Counts
from omni_eval.types.batch import Batch
from omni_eval.types.score import Score


class F1(Metric):
    name = "macro_f1"
    tasks = frozenset({"classification", "vision", "audio"})
    needs = frozenset({"label_truth", "label_pred"})
    level = "corpus"
    description = "F1 from full corpus confusion counts"

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        selected, points, failed = self.eligible(batch)
        selected = [c for c in selected if c.expected is not None and c.actual is not None]
        mode = self.name.split("_")[0] if "_" in self.name else "macro"
        counts = Counts(selected) if selected else None
        value = counts.aggregate("f1", mode) if counts else None
        return self._result(
            batch,
            [0.0] * len(selected),
            points,
            failed,
            value=value,
            details={"classes": counts.labels} if counts else {},
        )
