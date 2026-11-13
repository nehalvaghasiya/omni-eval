from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)
    text: str | None = None

    @model_validator(mode="after")
    def validate_span(self) -> "EvidenceSpan":
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("end must be >= start")

        return self
