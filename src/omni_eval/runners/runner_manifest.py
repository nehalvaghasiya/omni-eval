from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RunnerManifest(BaseModel):
    """Import contract; not an authorization to execute untrusted binaries."""

    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0"
    runner_id: str
    runner_version: str
    suite_id: str
    status: str
    metrics: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    findings: list[dict[str, Any]] = Field(default_factory=list)
    is_fixture: bool = False
