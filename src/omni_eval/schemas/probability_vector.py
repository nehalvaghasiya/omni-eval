import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class ProbabilityVector(BaseModel):
    """Ordered class probabilities; softmax must be a simplex."""

    model_config = ConfigDict(extra="forbid")
    labels: list[str]
    values: list[float]
    kind: Literal["softmax", "independent", "scores"] = "softmax"

    @model_validator(mode="after")
    def validate_vector(self) -> "ProbabilityVector":
        if (
            not self.labels
            or len(self.labels) != len(self.values)
            or len(set(self.labels)) != len(self.labels)
        ):
            raise ValueError("probabilities must have one value per unique class label")

        if any(not math.isfinite(x) for x in self.values):
            raise ValueError("probability values must be finite")

        if self.kind != "scores" and any(x < 0 or x > 1 for x in self.values):
            raise ValueError("probability values must be in [0,1]")

        if self.kind == "softmax" and not math.isclose(
            sum(self.values), 1.0, abs_tol=1e-5
        ):
            raise ValueError("softmax probabilities must sum to 1")

        return self
