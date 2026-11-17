"""Observed sandbox state must match expected state; agent claims do not count."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class StateSuccess(Metric):
    name = "state_success"
    tasks = frozenset({"agent", "mcp"})
    needs = frozenset({"trace", "expected.state"})

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        assert case.trace is not None and isinstance(case.expected, dict)
        if case.trace.state is None:
            raise ValueError("observed sandbox state missing")
        return float(all((case.trace.state.get(k) == v for k, v in case.expected["state"].items())))

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
