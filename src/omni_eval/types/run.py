"""Saved outcome; no metrics are recomputed for presentation."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from omni_eval.types.case import Case
from omni_eval.types.point import Point
from omni_eval.types.score import Score


class Run(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    status: str = "complete"
    task: str
    scores: list[Score]
    config: dict[str, Any] = Field(default_factory=dict)
    manifest: dict[str, Any] = Field(default_factory=dict)
    cases: list[Case] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    path: Path | None = None
    fixture: bool = False

    def metric(self, name: str) -> Score:
        for score in self.scores:
            if score.name == name:
                return score
        raise KeyError(f"metric {name!r} not found in run")

    def failures(self, metric: str | None = None) -> list[Point]:
        """Unavailable/error points and zero matches for higher-is-better metrics."""
        return [
            point
            for score in self.scores
            if metric is None or score.name == metric
            for point in score.points
            if point.status != "complete" or (score.direction == "up" and point.value == 0)
        ]

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "task": self.task,
            "fixture": self.fixture,
            "metrics": {s.name: s.value for s in self.scores},
            "coverage": {s.name: s.coverage for s in self.scores},
        }
