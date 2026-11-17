"""Mean share of selected contexts with annotated relevance."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.metrics.data import ranks, relevant
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class ContextCoverage(Metric):
    name = "context_precision"
    tasks = frozenset({"rag"})
    needs = frozenset({"contexts", "expected.relevant_ids"})

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        gold = relevant(case)
        assert gold is not None
        ranked = ranks(case)
        return len(gold & set(ranked)) / len(ranked) if ranked else 0

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
