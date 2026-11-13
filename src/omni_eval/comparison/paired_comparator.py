from __future__ import annotations

from functools import partial
from typing import Any

from omni_eval.core.evaluation_engine import _sample_metric
from omni_eval.core.metric_registry import MetricRegistry
from omni_eval.exceptions import InputError
from omni_eval.metrics.classification.classification_score import (
    calculate_classification,
)
from omni_eval.metrics.statistics.paired_bootstrap import paired_bootstrap_statistic
from omni_eval.schemas.comparison_result import ComparisonResult
from omni_eval.schemas.eval_sample import EvalSample
from omni_eval.schemas.prediction_record import PredictionRecord
from omni_eval.schemas.run_result import RunResult


def _metric_from_pairs(
    name: str, rows: list[tuple[EvalSample, PredictionRecord]], settings: dict[str, Any]
) -> float:
    descriptor = MetricRegistry().get(name)

    if descriptor.family == "classification":
        score, _ = calculate_classification(name, rows, settings)

        if score is None:
            raise InputError("non-scalar metric cannot be compared numerically")

        return score

    if descriptor.family == "performance":
        from omni_eval.metrics.performance.latency_percentiles import latency_percentile

        times = [pred.timing.latency_ms for _, pred in rows if pred.timing is not None]

        return (
            sum(times) / len(times)
            if name == "latency_mean"
            else latency_percentile(times, int(name.removeprefix("latency_p")))
        )

    return sum(
        _sample_metric(name, sample, pred, settings) for sample, pred in rows
    ) / len(rows)


def _paired_delta(
    indices: list[int],
    *,
    name: str,
    paired_rows: list[tuple[EvalSample, PredictionRecord, PredictionRecord]],
    settings: dict[str, Any],
) -> float:
    a = [(paired_rows[i][0], paired_rows[i][1]) for i in indices]
    b = [(paired_rows[i][0], paired_rows[i][2]) for i in indices]

    return _metric_from_pairs(name, b, settings) - _metric_from_pairs(name, a, settings)


class PairedComparator:
    """Paired comparison on shared, scored IDs with bootstrap uncertainty."""

    def compare(
        self,
        left: RunResult,
        right: RunResult,
        *,
        paired: bool = True,
        confidence: float = 0.95,
        seed: int = 42,
        iterations: int = 1000,
    ) -> list[ComparisonResult]:
        if not paired:
            raise InputError(
                "only paired comparison is implemented; unpaired scores cannot be presumed comparable"
            )

        if left.manifest.is_fixture != right.manifest.is_fixture:
            raise InputError("do not compare fixture results against real runs")

        if left.manifest.dataset_hash != right.manifest.dataset_hash:
            raise InputError("cannot compare different dataset fingerprints")

        if (
            not left.samples
            or not right.samples
            or not left.predictions
            or not right.predictions
        ):
            raise InputError(
                "comparison requires retained normalized samples and predictions"
            )

        common = {metric.name: metric for metric in left.metrics}.keys() & {
            metric.name: metric for metric in right.metrics
        }.keys()

        if not common:
            raise InputError("runs have no shared metrics")

        samples = {sample.sample_id: sample for sample in left.samples}
        left_predictions = {p.sample_id: p for p in left.predictions if not p.error}
        right_predictions = {p.sample_id: p for p in right.predictions if not p.error}
        results: list[ComparisonResult] = []

        for name in sorted(common):
            left_metric = next(m for m in left.metrics if m.name == name)
            right_metric = next(m for m in right.metrics if m.name == name)

            if left_metric.details.get("metric_version") != right_metric.details.get(
                "metric_version"
            ):
                raise InputError(f"metric version mismatch: {name}")

            if name.startswith("latency") and left_metric.details.get(
                "timing_provenance"
            ) != right_metric.details.get("timing_provenance"):
                results.append(
                    ComparisonResult(
                        left_run=left.manifest.run_id,
                        right_run=right.manifest.run_id,
                        metric=name,
                        paired_n=0,
                        delta=None,
                        status="incomparable",
                        details={"reason": "different timing measurement boundaries"},
                    )
                )
                continue

            left_configs = {
                m["name"]: m.get("settings", {})
                for m in left.manifest.metadata.get("metrics", [])
            }
            right_configs = {
                m["name"]: m.get("settings", {})
                for m in right.manifest.metadata.get("metrics", [])
            }

            if left_configs.get(name, {}) != right_configs.get(name, {}):
                raise InputError(f"metric settings mismatch: {name}")

            settings = left_configs.get(name, {})
            available = {
                s.sample_id
                for s in left.sample_scores
                if s.metric == name and s.status == "completed"
            }
            other = {
                s.sample_id
                for s in right.sample_scores
                if s.metric == name and s.status == "completed"
            }
            ids = sorted(
                (available & other) & left_predictions.keys() & right_predictions.keys()
            )

            # Global metrics often lack sample-level scores; fall back to shared non-error records.
            if not ids and left_metric.n_scored and right_metric.n_scored:
                ids = sorted(
                    left_predictions.keys() & right_predictions.keys() & samples.keys()
                )

            paired_rows = [
                (samples[sid], left_predictions[sid], right_predictions[sid])
                for sid in ids
            ]

            if not paired_rows:
                results.append(
                    ComparisonResult(
                        left_run=left.manifest.run_id,
                        right_run=right.manifest.run_id,
                        metric=name,
                        paired_n=0,
                        delta=None,
                        status="inconclusive",
                        details={"reason": "no eligible paired samples"},
                    )
                )
                continue

            try:
                delta = partial(
                    _paired_delta,
                    name=name,
                    paired_rows=paired_rows,
                    settings=settings,
                )
                estimate = delta(list(range(len(paired_rows))))
                interval = paired_bootstrap_statistic(
                    len(paired_rows),
                    delta,
                    confidence=confidence,
                    iterations=iterations,
                    seed=seed,
                )
            except (ValueError, TypeError, KeyError, InputError) as exc:
                results.append(
                    ComparisonResult(
                        left_run=left.manifest.run_id,
                        right_run=right.manifest.run_id,
                        metric=name,
                        paired_n=len(ids),
                        delta=None,
                        status="inconclusive",
                        details={"reason": str(exc)},
                    )
                )
                continue

            results.append(
                ComparisonResult(
                    left_run=left.manifest.run_id,
                    right_run=right.manifest.run_id,
                    metric=name,
                    paired_n=len(ids),
                    delta=estimate,
                    interval=interval,
                    details={
                        "left_only_n": len(left_predictions.keys() - set(ids)),
                        "right_only_n": len(right_predictions.keys() - set(ids)),
                        "confidence": confidence,
                        "seed": seed,
                        "iterations": iterations,
                    },
                )
            )

        return results
