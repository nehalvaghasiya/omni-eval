"""Structured judge answer, distinct from ground truth."""

from pydantic import BaseModel, ConfigDict, Field


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: float | None = Field(default=None, allow_inf_nan=False)
    status: str
    reason: str
    evidence: list[str] = Field(default_factory=list)
    revision: str
    tokens: int = Field(default=0, ge=0)
