"""Explicit judge credentials, network consent and finite call budgets."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Judge(BaseModel):
    """Only user-authorized OpenAI-compatible judges are supported."""

    model_config = ConfigDict(extra="forbid")
    provider: str = "openai_compatible"
    model: str
    rubric: str
    url_env: str
    token_env: str | None = None
    allowed_hosts: list[str] = Field(min_length=1)
    allowed_ports: list[int] = Field(default_factory=lambda: [443])
    local_http: bool = False
    max_calls: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    per_call_tokens: int = Field(default=512, gt=0)
    timeout: float = Field(default=20.0, gt=0, le=120)
    approve_external_data: bool = False

    @model_validator(mode="after")
    def validate_limits(self) -> "Judge":
        if self.per_call_tokens > self.max_tokens:
            raise ValueError("per_call_tokens cannot exceed total max_tokens")
        return self
