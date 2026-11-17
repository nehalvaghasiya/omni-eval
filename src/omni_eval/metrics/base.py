"""One metric abstraction; missing data and failures remain visible."""

from abc import ABC, abstractmethod
from typing import Any

from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.point import Point
from omni_eval.types.score import Score


class Metric(ABC):
    name: str = ""
    tasks: frozenset[str] = frozenset()
    needs: frozenset[str] = frozenset()
    direction: str = "up"
    version: str = "2"
    unit: str = "fraction"
    level: str = "sample"
    description: str = ""

    def __init__(self, *, k: int | None = None, name: str | None = None):
        self.k = k
        if name is not None:
            self.name = name

    def _has(self, case: Case, need: str) -> bool:
        if need == "actual":
            return case.actual is not None
        if need == "label_truth":
            from omni_eval.metrics.data import label

            return label(case, truth=True) is not None
        if need == "label_pred":
            from omni_eval.metrics.data import label

            return label(case) is not None
        if need == "expected":
            return case.expected is not None
        if need == "contexts":
            return bool(case.contexts)
        if need == "probs":
            return case.probs is not None
        if need == "trace":
            return case.trace is not None
        if need == "timing":
            return case.timing is not None
        if need == "expected.relevant_ids":
            return isinstance(case.expected, dict) and (
                "relevant_ids" in case.expected or "relevant_chunk_ids" in case.expected
            )
        if need == "expected.state":
            return isinstance(case.expected, dict) and "state" in case.expected
        return case.get(need) is not None

    def eligible(self, batch: Batch) -> tuple[list[Case], list[Point], int]:
        selected = []
        points = []
        failed = 0
        for case in batch.cases:
            if case.error is not None:
                points.append(
                    Point(
                        id=case.id,
                        metric=self.name,
                        status="error",
                        reason=case.error or "source reported an error",
                    )
                )
                failed += 1
                continue
            missing = [need for need in self.needs if not self._has(case, need)]
            if missing:
                status = (
                    "missing_reference"
                    if any(x.startswith("expected") or x == "label_truth" for x in missing)
                    else "missing_input"
                )
                points.append(
                    Point(
                        id=case.id,
                        metric=self.name,
                        status=status,
                        reason="Missing: " + ", ".join(sorted(missing)),
                    )
                )
            else:
                if "probs" in self.needs and "label_truth" in self.needs and case.probs is not None:
                    from omni_eval.metrics.data import label

                    if label(case, truth=True) not in case.probs.labels:
                        points.append(
                            Point(
                                id=case.id,
                                metric=self.name,
                                status="error",
                                reason="truth label not in probability vector",
                            )
                        )
                        failed += 1
                        continue
                selected.append(case)
        return (selected, points, failed)

    @abstractmethod
    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        """Return a fully counted metric, never a fake zero for unsupported input."""

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        raise NotImplementedError

    def sample_score(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        options = settings or {}
        selected, points, failed = self.eligible(batch)
        values = []
        for case in selected:
            try:
                value = float(self._point(case, options))
                from math import isfinite

                if not isfinite(value):
                    raise ValueError("nonfinite score")
                values.append(value)
                points.append(Point(id=case.id, metric=self.name, value=value))
            except (ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
                points.append(Point(id=case.id, metric=self.name, status="error", reason=str(exc)))
                failed += 1
        return self._result(batch, values, points, failed)

    def _result(
        self,
        batch: Batch,
        values: list[float],
        points: list[Point],
        failed: int,
        *,
        value: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> Score:
        total = len(batch.cases)
        scored = len(values)
        missing = sum(p.status.startswith("missing") for p in points)
        status = (
            "complete"
            if scored == total and total
            else "partial"
            if scored
            else "error"
            if failed
            else "missing_reference"
            if any(p.status == "missing_reference" for p in points)
            else "missing_input"
        )
        return Score(
            name=self.name,
            value=sum(values) / scored if scored and value is None else value,
            status=status,
            n_total=total,
            n_eligible=total - missing - failed,
            n_scored=scored,
            n_failed=failed,
            n_missing=missing,
            direction=self.direction,
            version=self.version,
            unit=self.unit,
            details=details or {},
            points=points,
        )
