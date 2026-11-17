"""One metric result with transparent denominator/coverage."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omni_eval.types.point import Point


class Score(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    value: float | None = Field(default=None, allow_inf_nan=False)
    status: str = "complete"
    n_total: int = Field(ge=0)
    n_eligible: int = Field(ge=0)
    n_scored: int = Field(ge=0)
    n_failed: int = Field(default=0, ge=0)
    n_missing: int = Field(default=0, ge=0)
    unit: str = "fraction"
    direction: str = "up"
    version: str = "1"
    source: str = "native"
    details: dict[str, Any] = Field(default_factory=dict)
    points: list[Point] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self) -> "Score":
        if self.n_scored > self.n_eligible or self.n_eligible > self.n_total:
            raise ValueError("invalid scoring counts")
        if self.value is not None and self.n_scored == 0 and (self.unit != "report"):
            raise ValueError("a numeric metric must have scored cases")
        return self

    @property
    def coverage(self) -> float:
        return self.n_scored / self.n_total if self.n_total else 0.0
