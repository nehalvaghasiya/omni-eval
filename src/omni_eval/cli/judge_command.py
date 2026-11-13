from pathlib import Path

import typer

from omni_eval.api import load_run, load_spec
from omni_eval.fixtures.fixture_predictions import classification_fixture
from omni_eval.judges.builtin_rubrics import get_rubric
from omni_eval.judges.fixture_judge_client import FixtureJudgeClient
from omni_eval.judges.judge_evaluator import JudgeEvaluator
from omni_eval.judges.openai_compatible_judge_client import OpenAICompatibleJudgeClient
from omni_eval.schemas.eval_sample import EvalSample
from omni_eval.schemas.prediction_record import PredictionRecord


def judge_command(
    run: Path | None = typer.Option(None, "--run"),
    config: Path | None = typer.Option(None, "--config"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    if dry_run:
        rubric = get_rubric("answer-correctness-v1")
        samples = [
            EvalSample(
                sample_id="fixture-judge",
                input={"question": "X"},
                reference={"answer": "X"},
            )
        ]
        predictions = [
            PredictionRecord(sample_id="fixture-judge", output={"answer": "X"})
        ]
        client = FixtureJudgeClient()
        max_calls = 1
    else:
        if run is None or config is None:
            raise ValueError("judge requires a saved --run and an explicit --config")

        saved = load_run(run)
        spec = load_spec(config)

        if spec.judge is None:
            raise ValueError(
                "config must include explicit judge selection and permissions"
            )

        rubric = get_rubric(spec.judge.rubric_id)
        client = (
            FixtureJudgeClient()
            if spec.judge.provider == "fixture"
            else OpenAICompatibleJudgeClient(spec.judge)
        )
        max_calls = spec.judge.max_calls
        samples, predictions = saved.samples, saved.predictions

    metric, scores = JudgeEvaluator(client, max_calls=max_calls).evaluate(
        samples, predictions, rubric
    )
    typer.echo(
        (
            "DRY RUN / SYNTHETIC — NOT A BENCHMARK\n"
            if dry_run or isinstance(client, FixtureJudgeClient)
            else ""
        )
        + f"{metric.name}: {metric.value} status={metric.status} n={metric.n_scored}/{metric.n_total}"
    )
