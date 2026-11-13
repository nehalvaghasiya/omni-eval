from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omni_eval.schemas.agent_step import AgentStep


class AgentTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: list[AgentStep] = Field(default_factory=list)
    final_output: Any = None
    observed_state: dict[str, Any] | None = None

    @model_validator(mode="after")
    def ordered_steps(self) -> "AgentTrace":
        indices = [step.index for step in self.steps]

        if indices != sorted(set(indices)):
            raise ValueError("agent step indices must be strictly ascending")

        return self
