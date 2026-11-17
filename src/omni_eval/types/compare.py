"""Paired comparison with compatibility notes."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CompareResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    baseline: str
    candidate: str
    deltas: dict[str, float | None] = Field(default_factory=dict)
    intervals: dict[str, list[float]] = Field(default_factory=dict)
    paired_n: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "candidate": self.candidate,
            "deltas": self.deltas,
            "intervals": self.intervals,
            "paired_n": self.paired_n,
        }
