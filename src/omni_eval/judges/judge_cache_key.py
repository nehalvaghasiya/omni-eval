from omni_eval.judges.judge_rubric import JudgeRubric
from omni_eval.schemas.eval_sample import EvalSample
from omni_eval.schemas.prediction_record import PredictionRecord
from omni_eval.sources.source_fingerprint import hash_canonical


class JudgeCacheKey:
    """Hash all inputs and rubric revision; never cache across model revisions."""

    def create(
        self,
        sample: EvalSample,
        prediction: PredictionRecord,
        rubric: JudgeRubric,
        model_revision: str,
    ) -> str:
        return hash_canonical(
            {
                "sample": sample.model_dump(mode="json"),
                "prediction": prediction.model_dump(mode="json"),
                "rubric": rubric.model_dump(mode="json"),
                "model_revision": model_revision,
            }
        )
