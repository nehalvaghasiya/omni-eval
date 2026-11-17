"""Observed tool success, distinct from final task success."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class ToolSuccess(Metric):
    name = "tool_success"
    tasks = frozenset({"agent", "mcp"})
    needs = frozenset({"trace"})

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        assert case.trace is not None
        calls = [s for s in case.trace.steps if s.kind == "tool_call"]
        if not calls:
            raise ValueError("no tool calls to score")
        if any(s.ok is None for s in calls):
            raise ValueError("tool call lacks observed ok status")
        return sum(s.ok is True for s in calls) / len(calls)

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
