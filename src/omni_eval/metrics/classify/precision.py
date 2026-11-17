"""Corpus precision macro/micro/weighted or explicit binary label."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.metrics.classify.stats import Counts
from omni_eval.types.batch import Batch
from omni_eval.types.score import Score


class Precision(Metric):
    name = "macro_precision"
    tasks = frozenset({"classification", "vision", "audio"})
    needs = frozenset({"label_truth", "label_pred"})
    level = "corpus"

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        options = settings or {}
        selected, points, failed = self.eligible(batch)
        counts = Counts(selected) if selected else None
        if counts is None:
            value = None
        elif self.name == "precision":
            positive = options.get("positive_label")
            if positive not in counts.labels:
                raise ValueError("precision requires declared positive_label")
            value = counts.rate(counts.labels.index(positive), "precision")
        else:
            value = counts.aggregate("precision", self.name.split("_")[0])
        return self._result(batch, [0.0] * len(selected), points, failed, value=value)
