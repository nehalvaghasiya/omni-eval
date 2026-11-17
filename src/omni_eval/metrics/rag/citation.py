"""Citation validity only checks IDs, not factual support."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.metrics.data import citations
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class CitationValid(Metric):
    name = "citation_valid"
    tasks = frozenset({"rag"})
    needs = frozenset({"actual", "contexts"})

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        cited = citations(case)
        if cited is None:
            raise ValueError("actual.citations list is required")
        ids = {c.id for c in case.contexts}
        return sum(x in ids for x in cited) / len(cited) if cited else 1.0

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
