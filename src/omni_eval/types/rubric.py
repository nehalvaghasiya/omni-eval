"""Versioned rubric definition."""

from pydantic import BaseModel, ConfigDict, model_validator


class Rubric(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    version: str
    criterion: str
    anchors: dict[str, str]
    required: list[str]
    low: float = 0
    high: float = 1
    require_evidence: bool = False

    @model_validator(mode="after")
    def validate_rubric(self) -> "Rubric":
        if not self.id or not self.version or not self.criterion.strip():
            raise ValueError("rubric needs id, version and a criterion")
        if self.low >= self.high:
            raise ValueError("rubric low must be smaller than high")
        if not self.anchors or not self.required:
            raise ValueError("rubric needs score anchors and required fields")
        allowed = {"input", "actual", "expected", "contexts", "trace"}
        if any(field.split(".", 1)[0] not in allowed for field in self.required):
            raise ValueError("rubric refers to a forbidden field")
        return self
