"""Per-case score or explicit unavailable reason."""

from pydantic import BaseModel, ConfigDict, Field


class Point(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    metric: str
    value: float | None = Field(default=None, allow_inf_nan=False)
    status: str = "complete"
    reason: str | None = None
    evidence: list[str] = Field(default_factory=list)
