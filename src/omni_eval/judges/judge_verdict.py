from pydantic import BaseModel, ConfigDict, Field


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: float | None
    rationale: str
    evidence: list[str] = Field(default_factory=list)
    abstained: bool = False
    fixture: bool = False
    model_revision: str
    latency_ms: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
