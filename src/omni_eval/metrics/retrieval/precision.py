"""Native precision retrieval scoring with explicit relevance judgments."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.metrics.retrieval.rank import Rank
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class PrecisionAtK(Metric):
    name = "precision@k"
    tasks = frozenset({"retrieval", "rag"})
    needs = frozenset({"expected.relevant_ids", "contexts"})

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        return Rank(case).metric("precision", self.k or 5)

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
