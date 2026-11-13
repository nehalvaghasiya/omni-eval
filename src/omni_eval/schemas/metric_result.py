from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MetricResult(BaseModel):
    """Aggregate and explicit denominator, with no imputed zero scores."""

    model_config = ConfigDict(extra="forbid")
    name: str
    value: float | None = None
    count: int = 0
    coverage: float = 0
    status: Literal[
        "completed",
        "partial",
        "insufficient_reference",
        "insufficient_input",
        "unsupported",
        "judge_error",
        "skipped",
    ] = "completed"
    n_total: int = 0
    n_eligible: int = 0
    n_scored: int = 0
    n_failed: int = 0
    n_missing: int = 0
    denominator: Literal["all_samples", "successful_only"] = "all_samples"
    details: dict[str, Any] = Field(default_factory=dict)
