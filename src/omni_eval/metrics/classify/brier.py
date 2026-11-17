"""Multiclass-vector Brier score, not single-positive Brier."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.metrics.data import label
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class Brier(Metric):
    name = "brier"
    needs = frozenset({"label_truth", "probs"})
    tasks = frozenset({"classification", "vision", "audio"})
    direction = "down"

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        prob = case.probs
        assert prob is not None
        if prob.kind != "softmax":
            raise ValueError("Brier needs softmax probabilities")
        truth = label(case, truth=True)
        if truth not in prob.labels:
            raise ValueError("truth label not in probability vector")
        return sum(
            ((p - float(x == truth)) ** 2 for x, p in zip(prob.labels, prob.values, strict=True))
        )

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
