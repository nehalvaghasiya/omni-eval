"""Repeated consecutive tool requests flag possible tool loops."""

import json
from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class LoopRate(Metric):
    name = "loop_rate"
    tasks = frozenset({"agent", "mcp"})
    needs = frozenset({"trace"})
    direction = "down"

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        assert case.trace is not None
        calls = [
            (x.name, json.dumps(x.args, sort_keys=True, default=str))
            for x in case.trace.steps
            if x.kind == "tool_call"
        ]
        return sum(calls[i] == calls[i - 1] for i in range(1, len(calls))) / max(1, len(calls) - 1)

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
