"""Explicit safe output shape checks with bounded regular expressions."""

import re
from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.metrics.data import answer
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class Format(Metric):
    name = "format"
    tasks = frozenset({"text", "rag", "agent"})
    needs = frozenset({"actual"})

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        actual = answer(case)
        if actual is None:
            raise ValueError("format requires text")
        if "max_length" in settings and len(actual) > int(settings["max_length"]):
            return 0.0
        if "min_length" in settings and len(actual) < int(settings["min_length"]):
            return 0.0
        pattern = settings.get("pattern")
        if pattern is not None:
            if not isinstance(pattern, str) or len(pattern) > 100:
                raise ValueError("format regex must be short")
            return float(re.fullmatch(pattern, actual) is not None)
        return 1.0

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
