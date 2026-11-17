"""Recorded reranking change, never model-generated qrels."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.metrics.retrieval.rank import Rank
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class RerankDelta(Metric):
    name = "rerank_delta"
    tasks = frozenset({"rag", "retrieval"})
    needs = frozenset({"expected.relevant_ids", "contexts"})

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        before = case.extensions.get("before")
        if not isinstance(before, list):
            raise ValueError("rerank_delta needs extensions.before ranked IDs")
        current = Rank(case)
        gold = current.truth
        k = self.k or 5
        from math import log2

        def ndcg(ids: list[str]) -> float:
            ideal = sum(1 / log2(i + 1) for i in range(1, min(k, len(gold)) + 1))
            return (
                sum((1 / log2(i + 1) for i, x in enumerate(ids[:k], 1) if x in gold)) / ideal
                if ideal
                else 1.0
            )

        return ndcg(current.items) - ndcg(before)

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
