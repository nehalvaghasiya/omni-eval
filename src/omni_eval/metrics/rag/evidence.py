"""Reference-annotated evidence retention, not judge faithfulness."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.metrics.data import ranks, relevant
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class Evidence(Metric):
    name = "evidence_coverage"
    tasks = frozenset({"rag"})
    needs = frozenset({"contexts", "expected.relevant_ids"})

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        gold = relevant(case)
        assert gold is not None
        return len(gold & set(ranks(case))) / len(gold) if gold else 1.0

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
