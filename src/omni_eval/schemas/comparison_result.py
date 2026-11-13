from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ComparisonResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    left_run: str
    right_run: str
    metric: str
    paired_n: int
    delta: float | None
    interval: tuple[float, float] | None = None
    status: str = "completed"
    details: dict[str, Any] = Field(default_factory=dict)
