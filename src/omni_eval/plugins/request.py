"""Versioned worker input contract."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WorkerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol: int = 1
    job: str
    provider: str
    operation: str
    metrics: list[str]
    cases: list[dict[str, Any]]
    limits: dict[str, Any] = Field(default_factory=dict)
