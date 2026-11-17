"""Top-k classification accuracy, ties by declared label order."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.metrics.data import label
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class TopK(Metric):
    name = "topk@k"
    needs = frozenset({"label_truth", "probs"})
    tasks = frozenset({"classification", "vision", "audio"})

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        prob = case.probs
        assert prob is not None
        k = self.k or 1
        if k > len(prob.labels):
            raise ValueError("k exceeds number of classes")
        order = sorted(range(len(prob.labels)), key=lambda i: (-prob.values[i], i))
        return float(label(case, truth=True) in [prob.labels[i] for i in order[:k]])

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
