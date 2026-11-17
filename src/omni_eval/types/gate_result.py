"""Three-valued gate verdict."""

from pydantic import BaseModel, ConfigDict


class GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rule: str
    status: str
    reason: str
    value: float | None = None
