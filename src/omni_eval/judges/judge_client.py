from typing import Protocol

from omni_eval.judges.judge_rubric import JudgeRubric
from omni_eval.judges.judge_verdict import JudgeVerdict
from omni_eval.schemas.eval_sample import EvalSample
from omni_eval.schemas.prediction_record import PredictionRecord


class JudgeClient(Protocol):
    def judge(
        self, sample: EvalSample, prediction: PredictionRecord, rubric: JudgeRubric
    ) -> JudgeVerdict: ...
