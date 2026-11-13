from omni_eval.judges.judge_rubric import JudgeRubric
from omni_eval.judges.judge_verdict import JudgeVerdict
from omni_eval.schemas.eval_sample import EvalSample
from omni_eval.schemas.prediction_record import PredictionRecord


class FixtureJudgeClient:
    """Deterministic synthetic verdict; never a real provider judgment."""

    def judge(
        self, sample: EvalSample, prediction: PredictionRecord, rubric: JudgeRubric
    ) -> JudgeVerdict:
        reference = (sample.reference or {}).get(
            "answer", (sample.reference or {}).get("text")
        )
        output = prediction.output.get("answer", prediction.output.get("text"))
        match = (
            isinstance(output, str)
            and reference is not None
            and output.strip() == str(reference).strip()
        )

        return JudgeVerdict(
            score=rubric.score_range[1] if match else rubric.score_range[0],
            rationale="Synthetic fixture exact-text comparison, not an LLM judge verdict",
            fixture=True,
            model_revision="fixture-v1",
        )
