"""Network target configuration never carries inline secret values."""

from pydantic import BaseModel, ConfigDict, Field


class Target(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str = "json"
    url_env: str
    token_env: str | None = None
    allowed_hosts: list[str] = Field(min_length=1)
    allowed_ports: list[int] = Field(default_factory=lambda: [443])
    model: str | None = None
    request: dict[str, str] = Field(default_factory=dict)
    response: dict[str, str] = Field(default_factory=dict)
    timeout: float = Field(default=20.0, gt=0, le=120)
    retries: int = Field(default=0, ge=0, le=3)
    concurrency: int = Field(default=1, ge=1, le=16)
    local_http: bool = False
