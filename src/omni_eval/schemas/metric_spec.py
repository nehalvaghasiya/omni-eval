from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MetricSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    settings: dict[str, Any] = Field(default_factory=dict)
    provider: str | None = None
