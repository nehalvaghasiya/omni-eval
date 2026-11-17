"""Observed agent execution trace."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omni_eval.types.step import Step


class Trace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: list[Step] = Field(default_factory=list)
    state: dict[str, Any] | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)
    actor: str | None = None

    @model_validator(mode="after")
    def validate_steps(self) -> "Trace":
        times = [s.time for s in self.steps if s.time is not None]
        if times != sorted(times):
            raise ValueError("agent trace timestamps must be ordered")
        return self
