"""Per-worker bounded resource settings."""

from pydantic import BaseModel, ConfigDict, Field


class Limits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_seconds: float = Field(default=120, gt=0)
    max_calls: int = Field(default=100, ge=0)
    max_bytes: int = Field(default=1000000, gt=0)
    allow_network: bool = False
