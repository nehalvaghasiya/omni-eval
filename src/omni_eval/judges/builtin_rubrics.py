"""Small explicit built-in rubrics; custom rubric files may be supplied later."""

from omni_eval.exceptions import InputError
from omni_eval.judges.judge_rubric import JudgeRubric


def get_rubric(rubric_id: str) -> JudgeRubric:
    definitions = {
        "answer-correctness-v1": (
            "judge_correctness",
            "Is the answer correct relative to the supplied reference?",
            ["output.answer", "reference.answer"],
        ),
        "groundedness-v1": (
            "judge_faithfulness",
            "Are all factual claims supported by the provided retrieved contexts?",
            ["output.answer", "contexts"],
        ),
        "answer-relevance-v1": (
            "judge_relevance",
            "Does the answer address the question in the input?",
            ["output.answer", "input.query"],
        ),
    }

    if rubric_id not in definitions:
        raise InputError(f"unknown judge rubric {rubric_id!r}")

    metric, criterion, required = definitions[rubric_id]

    return JudgeRubric(
        id=rubric_id,
        version=1,
        metric=metric,
        criterion=criterion,
        required_fields=required,
        score_range=(0, 1),
        anchors={"0": "Criterion not satisfied", "1": "Criterion fully satisfied"},
    )
