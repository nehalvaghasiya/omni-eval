from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0"
    run_id: str
    name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: Literal["completed", "partial", "invalid_input"] = "partial"
    source_mode: str
    dataset_hash: str
    predictions_hash: str
    config_hash: str
    package_version: str
    is_fixture: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
