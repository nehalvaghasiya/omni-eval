"""Reliable response rate uses all attempted cases as denominator."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.types.batch import Batch
from omni_eval.types.point import Point
from omni_eval.types.score import Score


class Reliability(Metric):
    name = "success_rate"
    level = "corpus"
    tasks = frozenset(
        {
            "classification",
            "text",
            "retrieval",
            "rag",
            "agent",
            "mcp",
            "security",
            "audio",
            "vision",
            "code",
            "sql",
        }
    )

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        total = len(batch.cases)
        success = sum(c.error is None and c.actual is not None for c in batch.cases)
        points = [
            Point(id=c.id, metric=self.name, value=float(c.error is None and c.actual is not None))
            for c in batch.cases
        ]
        return Score(
            name=self.name,
            value=success / total if total else None,
            status="complete" if total else "missing_input",
            n_total=total,
            n_eligible=total,
            n_scored=total,
            points=points,
            details={"policy": "all attempts, including timeout/errors"},
        )
