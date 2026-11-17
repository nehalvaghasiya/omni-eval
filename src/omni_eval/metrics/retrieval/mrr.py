"""Native mrr retrieval scoring with explicit relevance judgments."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.metrics.retrieval.rank import Rank
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class Mrr(Metric):
    name = "mrr"
    tasks = frozenset({"retrieval", "rag"})
    needs = frozenset({"expected.relevant_ids", "contexts"})

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        ranking = Rank(case)
        return ranking.metric("mrr", self.k or max(1, len(ranking.items)))

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
