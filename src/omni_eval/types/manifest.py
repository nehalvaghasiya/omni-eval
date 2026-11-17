"""Reproducibility manifest for saved evaluations."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = 1
    data_hash: str
    created: str
    package: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    fixture: bool = False
