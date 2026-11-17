"""Accuracy: correct labels divided by scored cases."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.metrics.data import label
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class Accuracy(Metric):
    name = "accuracy"
    tasks = frozenset({"classification", "vision", "audio"})
    needs = frozenset({"label_truth", "label_pred"})
    description = "Exact class label accuracy"

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        truth = label(case, truth=True)
        got = label(case)
        if truth is None or got is None:
            raise ValueError("label missing")
        return float(truth == got)

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
