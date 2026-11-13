from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SampleScore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_id: str
    metric: str
    value: float | None = None
    status: Literal[
        "completed",
        "insufficient_reference",
        "insufficient_input",
        "skipped",
        "target_error",
        "judge_error",
        "unsupported",
    ] = "completed"
    reason: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
