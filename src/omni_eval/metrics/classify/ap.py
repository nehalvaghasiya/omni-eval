"""Binary average precision; tie blocks share their threshold."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.metrics.data import label
from omni_eval.types.batch import Batch
from omni_eval.types.score import Score


class Ap(Metric):
    name = "auprc"
    needs = frozenset({"label_truth", "probs"})
    level = "corpus"
    tasks = frozenset({"classification", "vision", "audio"})

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        positive = (settings or {}).get("positive_label")
        if positive is None:
            raise ValueError("AUPRC requires positive_label")
        selected, points, failed = self.eligible(batch)
        data = []
        for c in selected:
            p = c.probs
            assert p is not None
            if positive not in p.labels:
                raise ValueError("positive label missing")
            data.append((p.values[p.labels.index(positive)], label(c, truth=True) == positive))
        total = sum((y for _, y in data))
        if total == 0:
            return self._result(
                batch, [], points, failed, details={"reason": "AUPRC needs a positive reference"}
            ).model_copy(update={"status": "unsupported"})
        data.sort(reverse=True, key=lambda x: x[0])
        seen = 0
        tp = 0
        area = 0.0
        i = 0
        while i < len(data):
            j = i + 1
            while j < len(data) and data[j][0] == data[i][0]:
                j += 1
            group = sum((y for _, y in data[i:j]))
            seen += j - i
            tp += group
            area += group / total * (tp / seen)
            i = j
        return self._result(batch, [0.0] * len(selected), points, failed, value=area)
