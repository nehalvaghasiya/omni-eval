"""Corpus recall macro/micro/weighted or explicit binary label."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.metrics.classify.stats import Counts
from omni_eval.types.batch import Batch
from omni_eval.types.score import Score


class Recall(Metric):
    name = "macro_recall"
    tasks = frozenset({"classification", "vision", "audio"})
    needs = frozenset({"label_truth", "label_pred"})
    level = "corpus"

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        options = settings or {}
        selected, points, failed = self.eligible(batch)
        counts = Counts(selected) if selected else None
        if counts is None:
            value = None
        elif self.name == "recall":
            positive = options.get("positive_label")
            if positive not in counts.labels:
                raise ValueError("recall requires positive_label")
            value = counts.rate(counts.labels.index(positive), "recall")
        else:
            value = counts.aggregate("recall", self.name.split("_")[0])
        return self._result(batch, [0.0] * len(selected), points, failed, value=value)
