"""One agent action or observation."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str
    name: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    ok: bool | None = None
    actor: str | None = None
    time: float | None = None
