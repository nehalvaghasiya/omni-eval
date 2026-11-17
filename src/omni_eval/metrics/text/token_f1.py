"""Token precision/recall/F1 from normalized word multisets."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.metrics.data import answer
from omni_eval.metrics.text.tokens import rates
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class TokenF1(Metric):
    name = "token_f1"
    tasks = frozenset({"text", "rag", "audio"})
    needs = frozenset({"expected", "actual"})

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        actual = answer(case)
        expected = answer(case, truth=True)
        if actual is None or expected is None:
            raise ValueError("requires text")
        index = {"token_precision": 0, "token_recall": 1, "token_f1": 2}[self.name]
        return rates(actual, expected)[index]

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
