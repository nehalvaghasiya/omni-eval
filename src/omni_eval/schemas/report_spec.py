from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ReportFormat = Literal["json", "html", "markdown", "junit"]


def _default_formats() -> list[ReportFormat]:
    return ["json", "html"]


class ReportSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    formats: list[ReportFormat] = Field(default_factory=_default_formats)
    retain_records: bool = True
