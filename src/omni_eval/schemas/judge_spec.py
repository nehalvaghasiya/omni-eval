from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class JudgeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["fixture", "openai_compatible"]
    model: str | None = None
    rubric_id: str
    temperature: float = Field(default=0, ge=0, le=2)
    max_calls: int = Field(default=100, ge=1)
    token_limit: int = Field(default=1024, ge=1)
    max_cost_usd: float = Field(default=1, ge=0)
    timeout_seconds: float = Field(default=20, gt=0, le=120)
    model_env: str | None = None
    base_url_env: str | None = None
    authorization_env: str | None = None
    data_use_approved: bool = False
    allowed_hosts: list[str] = Field(default_factory=list)
    allow_localhost: bool = False

    @model_validator(mode="after")
    def approve_network(self) -> "JudgeSpec":
        if self.provider != "fixture" and not self.data_use_approved:
            raise ValueError("external judge requires data_use_approved=true")

        return self
