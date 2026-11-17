"""Field-level exact key/value F1 for JSON objects."""

import json
from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class Fields(Metric):
    version = "2"
    name = "fields_f1"
    tasks = frozenset({"text", "rag", "agent"})
    needs = frozenset({"actual", "expected"})

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        if not isinstance(case.actual, dict) or not isinstance(case.expected, dict):
            raise ValueError("fields_f1 requires expected and actual objects")
        a = {(k, json.dumps(v, sort_keys=True, allow_nan=False)) for k, v in case.actual.items()}
        b = {(k, json.dumps(v, sort_keys=True, allow_nan=False)) for k, v in case.expected.items()}
        return 2 * len(a & b) / (len(a) + len(b)) if a or b else 1.0

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
