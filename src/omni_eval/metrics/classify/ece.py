"""Confidence-binned expected calibration error."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.metrics.data import label
from omni_eval.types.batch import Batch
from omni_eval.types.score import Score


class Ece(Metric):
    name = "ece"
    needs = frozenset({"label_truth", "probs"})
    direction = "down"
    level = "corpus"
    tasks = frozenset({"classification", "vision", "audio"})

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        bins = int((settings or {}).get("bins", 10))
        if not 2 <= bins <= 100:
            raise ValueError("calibration bins must be 2..100")
        selected, points, failed = self.eligible(batch)
        groups: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
        counted = 0
        for case in selected:
            prob = case.probs
            assert prob is not None
            if prob.kind != "softmax":
                from omni_eval.types.point import Point

                points.append(
                    Point(id=case.id, metric=self.name, status="error", reason="softmax required")
                )
                failed += 1
                continue
            index = max(range(len(prob.labels)), key=lambda i: prob.values[i])
            conf = prob.values[index]
            bucket = min(bins - 1, int(conf * bins))
            groups[bucket].append((conf, float(prob.labels[index] == label(case, truth=True))))
            counted += 1
        diffs = (
            [
                len(g)
                / counted
                * abs(sum((x for x, _ in g)) / len(g) - sum((y for _, y in g)) / len(g))
                for g in groups
                if g
            ]
            if counted
            else []
        )
        return self._result(
            batch,
            [0.0] * counted,
            points,
            failed,
            value=sum(diffs) if counted else None,
            details={"bins": bins, "bin_sizes": [len(g) for g in groups]},
        )
