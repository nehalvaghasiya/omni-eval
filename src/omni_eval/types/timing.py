"""Measured latency and provenance."""

from pydantic import BaseModel, ConfigDict, Field


class TimingData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    source: str = "measured_external"
    scope: str = "end_to_end"
    retries: int = Field(default=0, ge=0)
