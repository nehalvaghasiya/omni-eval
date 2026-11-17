"""Class-labelled probability vectors, never positional class guesses."""

from math import isfinite
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class Prob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    labels: list[str]
    values: list[float]
    kind: str = "softmax"

    @model_validator(mode="before")
    @classmethod
    def map_labels(cls, raw: Any) -> Any:
        # Beginner form: {"account": 0.8, "billing": 0.2}.
        # Sorting creates stable class order independent of JSON key order.
        if isinstance(raw, dict) and not any(
            field in raw for field in ("labels", "values", "kind")
        ):
            labels = sorted(raw)
            return {"labels": labels, "values": [raw[label] for label in labels]}
        return raw

    @model_validator(mode="after")
    def validate_prob(self) -> "Prob":
        if (
            not self.labels
            or len(self.labels) != len(self.values)
            or len(set(self.labels)) != len(self.labels)
        ):
            raise ValueError("probabilities need matching, unique, nonempty class labels")
        if self.kind not in {"softmax", "independent", "scores"}:
            raise ValueError("probability kind must be softmax, independent or scores")
        if any(not isfinite(v) for v in self.values):
            raise ValueError("probabilities must be finite")
        if self.kind != "scores" and any(v < 0 or v > 1 for v in self.values):
            raise ValueError("probabilities must be between zero and one")
        if self.kind == "softmax" and abs(sum(self.values) - 1) > 1e-05:
            raise ValueError("softmax probabilities must sum to one")
        return self
