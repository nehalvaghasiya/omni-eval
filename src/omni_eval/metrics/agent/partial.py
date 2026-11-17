"""Fraction of expected observed-state properties satisfied."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class StatePartial(Metric):
    name = "state_partial"
    tasks = frozenset({"agent", "mcp"})
    needs = frozenset({"trace", "expected.state"})

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        assert case.trace is not None and isinstance(case.expected, dict)
        if case.trace.state is None:
            raise ValueError("observed state missing")
        goal = case.expected["state"]
        return (
            sum((case.trace.state.get(k) == v for k, v in goal.items())) / len(goal)
            if goal
            else 1.0
        )

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
