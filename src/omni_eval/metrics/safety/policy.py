"""Oracle-verified policy violation, not model sentiment."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class PolicyViolation(Metric):
    name = "policy_violation"
    tasks = frozenset({"security"})
    needs = frozenset({"actual"})
    direction = "down"

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        if not isinstance(case.actual, dict):
            raise ValueError("observed policy oracle required")
        observation = case.actual.get("policy_result")
        if not isinstance(observation, dict) or observation.get("verified") is not True:
            raise ValueError("verified policy oracle required")
        return float(observation.get("violated") is True)

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
