from pydantic import BaseModel, ConfigDict, Field

from omni_eval.schemas.eval_sample import EvalSample
from omni_eval.schemas.metric_result import MetricResult
from omni_eval.schemas.prediction_record import PredictionRecord
from omni_eval.schemas.run_manifest import RunManifest
from omni_eval.schemas.sample_score import SampleScore


class RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    manifest: RunManifest
    metrics: list[MetricResult]
    sample_scores: list[SampleScore] = Field(default_factory=list)
    samples: list[EvalSample] = Field(default_factory=list)
    predictions: list[PredictionRecord] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    path: str | None = None
