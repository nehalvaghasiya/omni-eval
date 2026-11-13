from pydantic import BaseModel, ConfigDict, Field, model_validator


class JudgeRubric(BaseModel):
    """Immutable, explicit criterion; the model is not a ground-truth oracle."""

    model_config = ConfigDict(extra="forbid")
    id: str
    version: int = Field(ge=1)
    metric: str
    criterion: str
    required_fields: list[str]
    score_range: tuple[float, float]
    anchors: dict[str, str]
    allow_abstain: bool = True
    require_evidence: bool = True

    @model_validator(mode="after")
    def valid_anchors(self) -> "JudgeRubric":
        lower, upper = self.score_range

        if lower >= upper or not self.criterion.strip() or len(self.anchors) < 2:
            raise ValueError(
                "rubric requires criterion, ordered score range and >=2 anchors"
            )

        return self
