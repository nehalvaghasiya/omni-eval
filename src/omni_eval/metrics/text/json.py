"""Strict JSON syntax validity, including malformed generated strings."""

import json
from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class JsonValid(Metric):
    version = "2"
    name = "json_valid"
    tasks = frozenset({"text", "rag", "agent", "code"})
    needs = frozenset({"actual"})

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        value = case.actual
        if isinstance(value, str):
            try:
                json.loads(value, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            except (ValueError, json.JSONDecodeError):
                return 0.0
        else:
            try:
                json.dumps(value, allow_nan=False)
            except (TypeError, ValueError):
                return 0.0
        return 1.0

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
