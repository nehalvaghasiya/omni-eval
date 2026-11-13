from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvalSample(BaseModel):
    """Input and optional oracle data joined by stable sample_id."""

    model_config = ConfigDict(extra="forbid")
    sample_id: str
    input: dict[str, Any]
    reference: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    extensions: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sample_id")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("sample_id must not be empty")

        return value
