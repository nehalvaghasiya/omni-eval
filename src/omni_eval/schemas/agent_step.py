from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int = Field(ge=0)
    kind: str
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    result: Any = None
    actor_id: str | None = None
