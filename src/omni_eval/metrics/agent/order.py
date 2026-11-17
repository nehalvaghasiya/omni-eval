"""Required tool-order subsequence compliance."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class StepOrder(Metric):
    name = "step_order"
    tasks = frozenset({"agent", "mcp"})
    needs = frozenset({"trace", "expected"})

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        assert case.trace is not None
        required = case.expected.get("required_tools") if isinstance(case.expected, dict) else None
        if not isinstance(required, list):
            raise ValueError("expected.required_tools required")
        calls = iter(s.name for s in case.trace.steps if s.kind == "tool_call")
        return float(all(any(x == required_tool for x in calls) for required_tool in required))

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
