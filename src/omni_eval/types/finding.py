"""Scanner observations are not confirmed vulnerabilities."""

from pydantic import BaseModel, ConfigDict


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rule: str
    severity: str
    source: str
    verified: bool = False
    status: str = "observed"
    evidence: str | None = None
