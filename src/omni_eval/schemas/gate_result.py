from typing import Literal

from pydantic import BaseModel, ConfigDict


class GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gate_name: str
    status: Literal["pass", "fail", "inconclusive"]
    reason: str
    value: float | None = None
