"""Negative log probability for correct class."""

from math import log
from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.metrics.data import label
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class LogLoss(Metric):
    name = "logloss"
    needs = frozenset({"label_truth", "probs"})
    direction = "down"
    tasks = frozenset({"classification", "vision", "audio"})
    unit = "nats"

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        prob = case.probs
        assert prob is not None
        if prob.kind != "softmax":
            raise ValueError("logloss needs softmax")
        truth = label(case, truth=True)
        if truth not in prob.labels:
            raise ValueError("truth not in probability labels")
        return -log(max(1e-15, prob.values[prob.labels.index(truth)]))

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
