from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TargetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["http_json", "openai_compatible"] = "http_json"
    url: str | None = None
    url_env: str | None = None
    authorization_env: str | None = None
    method: Literal["POST"] = "POST"
    request_template: dict[str, Any] = Field(default_factory=dict)
    response_map: dict[str, str] = Field(default_factory=lambda: {"output": "$.result"})
    allowed_hosts: list[str] = Field(default_factory=list)
    allow_localhost: bool = False
    timeout_seconds: float = Field(default=20, gt=0, le=120)
    max_concurrency: int = Field(default=2, ge=1, le=16)
    retries: int = Field(default=0, ge=0, le=3)
    max_response_bytes: int = Field(default=1048576, ge=1024, le=10485760)

    @model_validator(mode="after")
    def require_location(self) -> "TargetSpec":
        if bool(self.url) == bool(self.url_env):
            raise ValueError("target requires exactly one of url or url_env")

        if not self.allowed_hosts:
            raise ValueError("target requires an explicit allowed_hosts list")

        return self
