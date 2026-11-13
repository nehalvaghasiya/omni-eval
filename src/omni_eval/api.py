"""Stable typed API shared by CLI and optional API server."""

from collections.abc import Iterable
from pathlib import Path

import yaml

from omni_eval.core.capability_registry import preflight
from omni_eval.core.evaluation_engine import EvaluationEngine
from omni_eval.exceptions import InputError
from omni_eval.fixtures.fixture_predictions import classification_fixture
from omni_eval.schemas.eval_sample import EvalSample
from omni_eval.schemas.metric_spec import MetricSpec
from omni_eval.schemas.prediction_record import PredictionRecord
from omni_eval.schemas.run_result import RunResult
from omni_eval.schemas.run_spec import RunSpec
from omni_eval.sources.jsonl_loader import load_jsonl
from omni_eval.storage.file_run_repository import FileRunRepository


def load_spec(path: str | Path) -> RunSpec:
    """Read the documented YAML/JSON run specification without executing templates."""
    try:
        with Path(path).open(encoding="utf-8") as source:
            config = yaml.safe_load(source)

        return RunSpec.model_validate(config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise InputError(f"invalid run specification {path}: {exc}") from exc


def evaluate(
    samples: str | Path | Iterable[EvalSample] | RunSpec | None = None,
    predictions: str | Path | Iterable[PredictionRecord] | None = None,
    metrics: list[str] | list[MetricSpec] | None = None,
    output_dir: str | Path | None = None,
    *,
    spec: RunSpec | None = None,
    dry_run: bool = False,
    strict: bool = False,
) -> RunResult:
    """Score saved records or an explicitly approved endpoint; no implicit model calls.

    A run with ``dry_run=True`` always uses synthetic fixtures and a separate
    temporary output root when writing, irrespective of any supplied target.
    """
    if isinstance(samples, RunSpec):
        spec = samples
        samples = None

    if spec is not None:
        if metrics is not None or predictions is not None:
            raise InputError("pass a RunSpec or individual fields, not both")

        dry_run = dry_run or spec.dry_run
        metrics_list = spec.metrics
    else:
        metrics_list = [
            m if isinstance(m, MetricSpec) else MetricSpec(name=m)
            for m in (metrics or ["accuracy"])
        ]

    judge_metrics = [
        metric for metric in metrics_list if metric.name.startswith("judge_")
    ]
    native_metrics = [
        metric for metric in metrics_list if not metric.name.startswith("judge_")
    ]

    if judge_metrics and (spec is None or spec.judge is None) and not dry_run:
        raise InputError(
            "judge metrics require an explicitly selected, approved RunSpec.judge"
        )

    if dry_run:
        records, outputs = classification_fixture()
        metrics_list = [
            m
            for m in metrics_list
            if m.name in {"accuracy", "macro_f1", "brier_score", "ece", "mcc"}
        ] or [MetricSpec(name="accuracy")]
        native_metrics = metrics_list
        judge_metrics = []
        mode = "fixture"
    elif spec is not None:
        if not spec.dataset:
            raise InputError("dataset path is required in non-fixture RunSpec")

        records = load_jsonl(spec.dataset, EvalSample)
        preflight(records, native_metrics)
        mode = spec.source.mode

        if mode == "offline":
            if spec.source.predictions is None:
                raise InputError("offline predictions path missing")

            outputs = load_jsonl(spec.source.predictions, PredictionRecord)
        elif mode == "replay":
            if spec.source.replay_path is None:
                raise InputError("replay path missing")

            old = FileRunRepository().load(spec.source.replay_path)
            from omni_eval.sources.source_fingerprint import hash_canonical

            dataset_hash = hash_canonical(
                [
                    s.model_dump(mode="json")
                    for s in sorted(records, key=lambda s: s.sample_id)
                ]
            )

            if old.manifest.dataset_hash != dataset_hash:
                raise InputError("replay dataset fingerprint mismatch")

            outputs = old.predictions
        elif mode in {"endpoint", "hybrid"}:
            from omni_eval.sources.http_prediction_source import HttpPredictionSource

            if spec.source.target is None:
                raise InputError("endpoint configuration missing")

            outputs = HttpPredictionSource(spec.source.target).acquire(records)
        else:
            raise InputError(f"unrecognized source mode {mode}")
    else:
        if samples is None or predictions is None:
            raise InputError(
                "provide samples and predictions, or a RunSpec (or --dry-run)"
            )

        records = (
            load_jsonl(samples, EvalSample)
            if isinstance(samples, (str, Path))
            else list(samples)
        )
        outputs = (
            load_jsonl(predictions, PredictionRecord)
            if isinstance(predictions, (str, Path))
            else list(predictions)
        )
        mode = "offline"

    run = EvaluationEngine().evaluate(
        records,
        outputs,
        native_metrics,
        name=spec.name
        if spec and not dry_run
        else "DRY RUN / SYNTHETIC"
        if dry_run
        else "evaluation",
        source_mode=mode,
        is_fixture=dry_run,
        strict=strict,
        config=spec.model_dump(mode="json") if spec and not dry_run else None,
    )

    if judge_metrics and spec is not None and spec.judge is not None:
        from omni_eval.judges.builtin_rubrics import get_rubric
        from omni_eval.judges.fixture_judge_client import FixtureJudgeClient
        from omni_eval.judges.judge_evaluator import JudgeEvaluator
        from omni_eval.judges.openai_compatible_judge_client import (
            OpenAICompatibleJudgeClient,
        )

        rubric = get_rubric(spec.judge.rubric_id)

        if {metric.name for metric in judge_metrics} != {rubric.metric}:
            raise InputError(
                "selected judge metrics do not match the configured rubric"
            )

        client = (
            FixtureJudgeClient()
            if spec.judge.provider == "fixture"
            else OpenAICompatibleJudgeClient(spec.judge)
        )
        judge_metric, judge_scores = JudgeEvaluator(
            client, max_calls=spec.judge.max_calls
        ).evaluate(records, outputs, rubric)
        run.metrics.append(judge_metric)
        run.sample_scores.extend(judge_scores)

        if judge_metric.status != "completed":
            run.manifest.status = "partial"

        if isinstance(client, FixtureJudgeClient):
            run.manifest.is_fixture = True
            run.manifest.metadata["marker"] = "DRY RUN / SYNTHETIC — FIXTURE JUDGE"

        run.manifest.metadata["rubric"] = rubric.model_dump(mode="json")

    if mode == "hybrid" and spec is not None and spec.source.predictions:
        baseline_predictions = load_jsonl(spec.source.predictions, PredictionRecord)
        baseline_run = EvaluationEngine().evaluate(
            records,
            baseline_predictions,
            native_metrics,
            name=spec.name + " baseline",
            source_mode="offline",
        )
        from omni_eval.comparison.paired_comparator import PairedComparator

        comparisons = PairedComparator().compare(baseline_run, run, iterations=1000)
        run.manifest.metadata["hybrid_baseline_hash"] = (
            baseline_run.manifest.predictions_hash
        )
        run.manifest.metadata["hybrid_comparisons"] = [
            result.model_dump(mode="json") for result in comparisons
        ]

    if output_dir is not None:
        FileRunRepository().save(
            run,
            output_dir,
            formats=tuple(spec.report.formats) if spec else ("json", "html"),
            retain_records=spec.report.retain_records if spec else True,
        )

    return run


def load_run(path: str | Path) -> RunResult:
    """Load a saved immutable run, verifying every artifact hash."""
    return FileRunRepository().load(path)


def compare(
    baseline: RunResult | str | Path,
    candidate: RunResult | str | Path,
    *,
    paired: bool = True,
    confidence: float = 0.95,
    seed: int = 42,
    iterations: int = 1000,
) -> list["ComparisonResult"]:
    """Compare compatible saved runs with paired bootstrap intervals."""
    from omni_eval.comparison.paired_comparator import PairedComparator
    from omni_eval.schemas.comparison_result import ComparisonResult

    left = load_run(baseline) if isinstance(baseline, (str, Path)) else baseline
    right = load_run(candidate) if isinstance(candidate, (str, Path)) else candidate

    return PairedComparator().compare(
        left,
        right,
        paired=paired,
        confidence=confidence,
        seed=seed,
        iterations=iterations,
    )
