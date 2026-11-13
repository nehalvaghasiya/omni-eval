from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ReportSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    formats: list[Literal["json", "html", "markdown", "junit"]] = Field(
        default_factory=lambda: ["json", "html"]
    )
    retain_records: bool = True
