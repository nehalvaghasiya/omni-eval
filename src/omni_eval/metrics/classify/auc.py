"""Binary rank AUC with average ties, no optional sklearn dependency."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.metrics.data import label
from omni_eval.types.batch import Batch
from omni_eval.types.score import Score


class Auc(Metric):
    name = "auroc"
    needs = frozenset({"label_truth", "probs"})
    level = "corpus"
    tasks = frozenset({"classification", "vision", "audio"})

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        positive = (settings or {}).get("positive_label")
        if positive is None:
            raise ValueError("AUROC requires positive_label setting")
        selected, points, failed = self.eligible(batch)
        data = []
        for case in selected:
            p = case.probs
            assert p is not None
            if positive not in p.labels:
                raise ValueError("positive_label absent from probabilities")
            data.append((p.values[p.labels.index(positive)], label(case, truth=True) == positive))
        pos = sum((y for _, y in data))
        neg = len(data) - pos
        if not pos or not neg:
            return self._result(
                batch, [], points, failed, details={"reason": "AUROC needs both reference classes"}
            ).model_copy(update={"status": "unsupported"})
        wins = sum(
            (
                1 if score_pos > score_neg else 0.5 if score_pos == score_neg else 0
                for score_pos, y in data
                if y
                for score_neg, z in data
                if not z
            )
        )
        return self._result(batch, [0.0] * len(selected), points, failed, value=wins / (pos * neg))
