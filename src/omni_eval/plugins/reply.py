"""Versioned worker result; only finite scores accepted."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WorkerReply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol: int = 1
    job: str
    provider: str
    status: str
    metrics: list[dict[str, Any]]
    errors: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_version(self) -> "WorkerReply":
        if self.protocol != 1:
            raise ValueError("unsupported worker protocol")
        if self.status not in {"complete", "partial", "error"}:
            raise ValueError("invalid worker status")
        return self
