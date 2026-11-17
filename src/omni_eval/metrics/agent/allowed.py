"""Tool allowlist compliance on observed calls."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class ToolAllowed(Metric):
    name = "tool_allowed"
    tasks = frozenset({"agent", "mcp"})
    needs = frozenset({"trace", "expected"})

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        assert case.trace is not None
        allowed = case.expected.get("allowed_tools") if isinstance(case.expected, dict) else None
        if not isinstance(allowed, list):
            raise ValueError("expected.allowed_tools required")
        calls = [s for s in case.trace.steps if s.kind == "tool_call"]
        return sum(s.name in allowed for s in calls) / len(calls) if calls else 1.0

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
