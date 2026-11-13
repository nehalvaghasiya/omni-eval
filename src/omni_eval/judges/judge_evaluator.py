from omni_eval.exceptions import AcquisitionError, InputError
from omni_eval.judges.judge_client import JudgeClient
from omni_eval.judges.judge_rubric import JudgeRubric
from omni_eval.schemas.eval_sample import EvalSample
from omni_eval.schemas.metric_result import MetricResult
from omni_eval.schemas.prediction_record import PredictionRecord
from omni_eval.schemas.sample_score import SampleScore


class JudgeEvaluator:
    """Budget-limited call count and structured sample diagnostic aggregation."""

    def __init__(self, client: JudgeClient, max_calls: int):
        self.client = client
        self.max_calls = max_calls

    def evaluate(
        self,
        samples: list[EvalSample],
        predictions: list[PredictionRecord],
        rubric: JudgeRubric,
    ) -> tuple[MetricResult, list[SampleScore]]:
        by_id = {p.sample_id: p for p in predictions}
        scores: list[SampleScore] = []
        values: list[float] = []
        calls = 0

        for sample in samples:
            prediction = by_id.get(sample.sample_id)

            if prediction is None or prediction.error:
                scores.append(
                    SampleScore(
                        sample_id=sample.sample_id,
                        metric=rubric.metric,
                        status="skipped",
                        reason="prediction missing/failed",
                    )
                )
                continue

            missing = [
                path
                for path in rubric.required_fields
                if not self._exists(path, sample, prediction)
            ]

            if missing:
                scores.append(
                    SampleScore(
                        sample_id=sample.sample_id,
                        metric=rubric.metric,
                        status="insufficient_reference",
                        reason=f"missing {missing}",
                    )
                )
                continue

            if calls >= self.max_calls:
                scores.append(
                    SampleScore(
                        sample_id=sample.sample_id,
                        metric=rubric.metric,
                        status="skipped",
                        reason="judge call budget exhausted",
                    )
                )
                continue

            calls += 1

            try:
                verdict = self.client.judge(sample, prediction, rubric)

                if verdict.abstained or verdict.score is None:
                    raise AcquisitionError("judge abstained")

                if not rubric.score_range[0] <= verdict.score <= rubric.score_range[1]:
                    raise AcquisitionError("judge score out of bounds")

                scores.append(
                    SampleScore(
                        sample_id=sample.sample_id,
                        metric=rubric.metric,
                        value=verdict.score,
                        evidence={
                            "fixture": verdict.fixture,
                            "reason": verdict.rationale,
                            "evidence": verdict.evidence,
                            "model_revision": verdict.model_revision,
                        },
                    )
                )
                values.append(verdict.score)
            except AcquisitionError as exc:
                scores.append(
                    SampleScore(
                        sample_id=sample.sample_id,
                        metric=rubric.metric,
                        status="judge_error",
                        reason=str(exc),
                    )
                )

        n_total = len(samples)
        n_scored = len(values)
        metric = MetricResult(
            name=rubric.metric,
            value=sum(values) / len(values) if values else None,
            n_total=n_total,
            n_eligible=n_total,
            n_scored=n_scored,
            count=n_scored,
            coverage=n_scored / n_total if n_total else 0,
            n_failed=calls - n_scored,
            n_missing=sum(p.sample_id not in by_id for p in samples),
            status="completed"
            if n_scored == n_total
            else "partial"
            if n_scored
            else "judge_error",
            details={
                "rubric_id": rubric.id,
                "rubric_version": rubric.version,
                "judge_calls": calls,
                "judge_provenance": type(self.client).__name__,
            },
        )

        return metric, scores

    @staticmethod
    def _exists(path: str, sample: EvalSample, prediction: PredictionRecord) -> bool:
        from omni_eval.sources.field_mapper import get_path

        payload = {
            "input": sample.input,
            "reference": sample.reference,
            "output": prediction.output,
            "contexts": [c.model_dump() for c in prediction.contexts or []],
        }

        try:
            value = get_path(payload, "$." + path)

            return value is not None and (not isinstance(value, list) or len(value) > 0)
        except InputError:
            return False
