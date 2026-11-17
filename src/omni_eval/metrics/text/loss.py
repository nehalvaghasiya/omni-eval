"""Token negative log likelihood needs explicit token log probabilities."""

from math import exp, isfinite
from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.types.batch import Batch
from omni_eval.types.point import Point
from omni_eval.types.score import Score


class Loss(Metric):
    version = "2"
    name = "nll"
    tasks = frozenset({"text"})
    needs = frozenset({"actual"})
    direction = "down"
    level = "corpus"
    unit = "nats"

    def __init__(self, *, k: int | None = None, name: str | None = None):
        super().__init__(k=k, name=name)
        self.unit = "ratio" if self.name == "perplexity" else "nats"

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        selected, points, failed = self.eligible(batch)
        logps: list[float] = []
        count = 0
        for c in selected:
            if not isinstance(c.actual, dict) or not isinstance(c.actual.get("logprobs"), list):
                points.append(
                    Point(
                        id=c.id,
                        metric=self.name,
                        status="missing_input",
                        reason="token logprobs required",
                    )
                )
                continue
            values = c.actual["logprobs"]
            if not values:
                points.append(
                    Point(
                        id=c.id,
                        metric=self.name,
                        status="missing_input",
                        reason="token logprobs are empty",
                    )
                )
                continue
            if any(type(v) not in (int, float) or not isfinite(v) or v > 0 for v in values):
                points.append(
                    Point(
                        id=c.id,
                        metric=self.name,
                        status="error",
                        reason="logprobs must be finite nonpositive numbers",
                    )
                )
                failed += 1
                continue
            logps.extend(values)
            count += 1
        value = -sum(logps) / len(logps) if logps else None
        if value is not None and self.name == "perplexity":
            try:
                value = exp(value)
            except OverflowError:
                return self._result(
                    batch,
                    [],
                    points,
                    failed,
                    details={"reason": "perplexity exceeds finite float range"},
                ).model_copy(update={"status": "unsupported"})
        return self._result(
            batch,
            [0.0] * count,
            points,
            failed,
            value=value,
            details={"tokens": len(logps), "normalization": "per token"},
        )
