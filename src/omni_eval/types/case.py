"""One stable-ID reference/prediction pair."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omni_eval.types.context import Context
from omni_eval.types.prob import Prob
from omni_eval.types.timing import TimingData
from omni_eval.types.trace import Trace


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    input: Any = None
    expected: Any = None
    actual: Any = None
    probs: Prob | None = None
    contexts: list[Context] = Field(default_factory=list)
    trace: Trace | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    timing: TimingData | None = None
    error: str | None = None
    extensions: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("id must be nonempty")
        return value

    @model_validator(mode="after")
    def unique_contexts(self) -> "Case":
        ids = [c.id for c in self.contexts]
        ranks = [c.rank for c in self.contexts]
        if len(set(ids)) != len(ids) or len(set(ranks)) != len(ranks):
            raise ValueError("context ids and ranks must be unique")
        return self

    def get(self, path: str) -> Any:
        """Safely read a dotted canonical field; no eval/attribute traversal."""
        obj: Any = self.model_dump(mode="python")
        for piece in path.split("."):
            if isinstance(obj, dict):
                obj = obj.get(piece)
            elif isinstance(obj, list) and piece.isdigit():
                index = int(piece)
                obj = obj[index] if index < len(obj) else None
            else:
                return None
        return obj
