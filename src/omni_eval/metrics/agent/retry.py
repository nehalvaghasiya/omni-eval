"""Observed retry event rate per tool call."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class RetryRate(Metric):
    name = "retry_rate"
    tasks = frozenset({"agent", "mcp"})
    needs = frozenset({"trace"})
    direction = "down"

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        assert case.trace is not None
        calls = sum(s.kind == "tool_call" for s in case.trace.steps)
        retry = sum(s.kind == "retry" for s in case.trace.steps)
        return retry / max(1, calls)

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
