from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RagTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stages: dict[str, Any] = Field(default_factory=dict)
    answer: str | None = None
    citation_ids: list[str] = Field(default_factory=list)
