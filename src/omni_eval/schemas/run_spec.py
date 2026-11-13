from pydantic import BaseModel, ConfigDict, Field

from omni_eval.schemas.judge_spec import JudgeSpec
from omni_eval.schemas.metric_spec import MetricSpec
from omni_eval.schemas.report_spec import ReportSpec
from omni_eval.schemas.source_spec import SourceSpec


class RunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0"
    name: str = "omni-eval-run"
    dataset: str | None = None
    source: SourceSpec
    metrics: list[MetricSpec]
    report: ReportSpec = Field(default_factory=ReportSpec)
    judge: JudgeSpec | None = None
    dry_run: bool = False

    def model_post_init(self, __context: object) -> None:
        if self.schema_version != "1.0":
            raise ValueError("only RunSpec schema version 1.0 is supported")

        if not self.metrics:
            raise ValueError("a run requires at least one metric")
