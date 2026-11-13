from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SecurityFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: str
    severity: Literal["info", "low", "medium", "high", "critical"]
    status: Literal["observed", "verified", "unverified", "false_positive"]
    evidence_refs: list[str] = Field(default_factory=list)
    scanner: str
