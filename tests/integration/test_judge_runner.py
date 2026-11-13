import json
from pathlib import Path

import pytest

from omni_eval.adapters.runner_result_importer import RunnerResultImporter
from omni_eval.exceptions import InputError
from omni_eval.fixtures.fixture_predictions import classification_fixture
from omni_eval.judges.builtin_rubrics import get_rubric
from omni_eval.judges.fixture_judge_client import FixtureJudgeClient
from omni_eval.judges.judge_cache_key import JudgeCacheKey
from omni_eval.judges.judge_evaluator import JudgeEvaluator
from omni_eval.runners.fixture_runner import FixtureRunner
from omni_eval.schemas.eval_sample import EvalSample
from omni_eval.schemas.prediction_record import PredictionRecord


def test_judge_fixture_is_declared_synthetic():
    rubric = get_rubric("answer-correctness-v1")
    sample = EvalSample(sample_id="x", input={}, reference={"answer": "a"})
    pred = PredictionRecord(sample_id="x", output={"answer": "a"})
    verdict = FixtureJudgeClient().judge(sample, pred, rubric)

    assert verdict.fixture
    metric, scores = JudgeEvaluator(FixtureJudgeClient(), max_calls=1).evaluate(
        [sample], [pred], rubric
    )

    assert metric.value == 1
    assert scores[0].evidence["fixture"] is True
    assert JudgeCacheKey().create(
        sample, pred, rubric, "revision-a"
    ) != JudgeCacheKey().create(sample, pred, rubric, "revision-b")


def test_judge_budget_records_partial():
    rubric = get_rubric("answer-correctness-v1")
    samples = [
        EvalSample(sample_id=str(i), input={}, reference={"answer": "a"})
        for i in range(3)
    ]
    preds = [
        PredictionRecord(sample_id=str(i), output={"answer": "a"}) for i in range(3)
    ]
    metric, _ = JudgeEvaluator(FixtureJudgeClient(), max_calls=1).evaluate(
        samples, preds, rubric
    )

    assert metric.n_scored == 1 and metric.coverage == pytest.approx(1 / 3)


def test_runner_import_requires_versioned_result(tmp_path):
    fixture = FixtureRunner().execute()
    file = tmp_path / "runner-result.json"
    file.write_text(fixture.model_dump_json())

    assert RunnerResultImporter().import_file(file).is_fixture
    file.write_text(json.dumps({"runner_id": "unversioned"}))

    with pytest.raises(InputError):
        RunnerResultImporter().import_file(file)
