"""Exact case-sensitive and normalized answer equality."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.metrics.data import answer
from omni_eval.metrics.text.normalize import normalize
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class Exact(Metric):
    name = "exact"
    tasks = frozenset({"text", "rag", "audio", "code", "sql"})
    needs = frozenset({"expected", "actual"})

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        actual = answer(case)
        expected = answer(case, truth=True)
        if actual is None or expected is None:
            raise ValueError("expected and actual text required")
        if self.name == "answer_match":
            return float(normalize(actual) == normalize(expected))
        return float(actual == expected)

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
