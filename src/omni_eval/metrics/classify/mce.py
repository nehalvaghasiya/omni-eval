"""Maximum confidence-bin calibration discrepancy."""

from typing import Any

from omni_eval.metrics.classify.ece import Ece
from omni_eval.metrics.data import label
from omni_eval.types.batch import Batch
from omni_eval.types.point import Point
from omni_eval.types.score import Score


class Mce(Ece):
    version = "2"
    name = "mce"

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
                points.append(
                    Point(id=case.id, metric=self.name, status="error", reason="softmax required")
                )
                failed += 1
                continue
            index = max(range(len(prob.labels)), key=lambda i: prob.values[i])
            conf = prob.values[index]
            groups[min(bins - 1, int(conf * bins))].append(
                (conf, float(prob.labels[index] == label(case, truth=True)))
            )
            counted += 1
        gaps = [
            abs(sum((x for x, _ in g)) / len(g) - sum((y for _, y in g)) / len(g))
            for g in groups
            if g
        ]
        return self._result(
            batch,
            [0.0] * counted,
            points,
            failed,
            value=max(gaps) if gaps else None,
            details={"bins": bins},
        )
