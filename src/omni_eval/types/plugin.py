"""Explicit plugin operation and capability."""

from pydantic import BaseModel, ConfigDict


class PluginSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    operation: str = "score"
    suite: str | None = None
