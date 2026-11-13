from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omni_eval.schemas.agent_trace import AgentTrace
from omni_eval.schemas.probability_vector import ProbabilityVector
from omni_eval.schemas.rag_trace import RagTrace
from omni_eval.schemas.retrieved_item import RetrievedItem
from omni_eval.schemas.timing_record import TimingRecord


class PredictionRecord(BaseModel):
    """Canonical prediction independent of the acquisition mechanism."""

    model_config = ConfigDict(extra="forbid")
    sample_id: str
    output: dict[str, Any]
    probabilities: ProbabilityVector | None = None
    contexts: list[RetrievedItem] | None = None
    trace: AgentTrace | None = None
    rag_trace: RagTrace | None = None
    timing: TimingRecord | None = None
    error: str | None = None
    extensions: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sample_id")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("sample_id must not be empty")

        return value

    @model_validator(mode="after")
    def check_contexts(self) -> "PredictionRecord":
        if self.contexts is not None:
            ranks = [item.rank for item in self.contexts]
            ids = [item.chunk_id for item in self.contexts]

            if len(ranks) != len(set(ranks)) or len(ids) != len(set(ids)):
                raise ValueError("contexts must have unique ranks and chunk IDs")

        return self
