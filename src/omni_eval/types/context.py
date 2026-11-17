"""Ranked retrieved document/chunk."""

from pydantic import BaseModel, ConfigDict, Field


class Context(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    text: str | None = None
    score: float | None = None
    source: str | None = None
