"""Conservative read-only SQL check, not SQL execution."""

import re
from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.metrics.data import answer
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class Parse(Metric):
    name = "sql_select_safe"
    tasks = frozenset({"sql"})
    needs = frozenset({"actual"})

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        sql = answer(case)
        if sql is None:
            raise ValueError("SQL text required")
        cleaned = sql.strip()
        if not re.match("(?is)^select\\s+", cleaned):
            return 0.0
        if ";" in cleaned.rstrip(";") or re.search(
            "(?is)\\b(insert|update|delete|drop|alter|attach|pragma|create|exec)\\b", cleaned
        ):
            return 0.0
        return float(cleaned.endswith(";") or bool(cleaned))

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
