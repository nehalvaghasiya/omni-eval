"""Explicit numeric/coverage regression gate."""

from pydantic import BaseModel, ConfigDict, Field


class GateRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric: str
    op: str = "gte"
    value: float = Field(allow_inf_nan=False)
    min_coverage: float = Field(default=0.0, ge=0, le=1)
    min_pairs: int = Field(default=0, ge=0)
