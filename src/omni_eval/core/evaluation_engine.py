"""Canonical evaluation. No model/network code is imported by this module."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import numpy as np

from omni_eval.core.capability_registry import has_reference, preflight
from omni_eval.core.metric_registry import MetricRegistry
from omni_eval.core.sample_aligner import SampleAligner
from omni_eval.exceptions import InputError
from omni_eval.metrics.agent.permission_violation import unauthorized_tool_rate
from omni_eval.metrics.agent.state_success import state_success
from omni_eval.metrics.agent.trajectory_check import trajectory_check
from omni_eval.metrics.classification.classification_score import (
    calculate_classification,
)
from omni_eval.metrics.performance.latency_percentiles import latency_percentile
from omni_eval.metrics.rag.citation_validity import citation_validity
from omni_eval.metrics.retrieval.ranking import ranking_metric
from omni_eval.metrics.text.exact_match import exact_match
from omni_eval.metrics.text.json_validity import json_validity
from omni_eval.metrics.text.schema_compliance import schema_compliance
from omni_eval.metrics.text.token_f1 import token_scores
from omni_eval.metrics.text.word_error_rate import error_rate
from omni_eval.schemas.eval_sample import EvalSample
from omni_eval.schemas.metric_result import MetricResult
from omni_eval.schemas.metric_spec import MetricSpec
from omni_eval.schemas.prediction_record import PredictionRecord
from omni_eval.schemas.run_manifest import RunManifest
from omni_eval.schemas.run_result import RunResult
from omni_eval.schemas.sample_score import SampleScore
from omni_eval.sources.source_fingerprint import hash_canonical


def _output_text(prediction: PredictionRecord) -> str | None:
    value = prediction.output.get("answer", prediction.output.get("text"))

    return value if isinstance(value, str) else None


def _reference_text(sample: EvalSample) -> str:
    ref = sample.reference or {}

    return str(next(ref[key] for key in ("answer", "text", "value") if key in ref))


def _has_prediction(prediction: PredictionRecord, family: str, required: str) -> bool:
    if prediction.error:
        return False

    if required == "output":
        return True

    if required == "label":
        return "label" in prediction.output

    if required == "probabilities":
        return prediction.probabilities is not None and "label" in prediction.output

    if required == "text":
        return _output_text(prediction) is not None

    if required == "contexts":
        return prediction.contexts is not None

    if required == "trace":
        return prediction.trace is not None

    if required == "timing":
        return prediction.timing is not None

    if required == "oracle_result":
        return "oracle_result" in prediction.output

    return False


def _sample_metric(
    name: str, sample: EvalSample, pred: PredictionRecord, settings: dict[str, Any]
) -> float:
    ref = sample.reference or {}

    if name in ("exact_match", "normalized_exact_match"):
        return exact_match(
            _output_text(pred) or "",
            _reference_text(sample),
            normalized=name.startswith("normalized"),
        )

    if name in ("token_precision", "token_recall", "token_f1"):
        return dict(
            zip(
                ("token_precision", "token_recall", "token_f1"),
                token_scores(_output_text(pred) or "", _reference_text(sample)),
                strict=True,
            )
        )[name]

    if name == "json_validity":
        return json_validity(
            pred.output.get("text", pred.output.get("answer", pred.output))
        )

    if name == "json_schema_validity":
        schema = ref.get("schema", sample.input.get("schema"))

        if not isinstance(schema, dict):
            raise ValueError("JSON schema must be an object")

        return schema_compliance(
            pred.output.get("text", pred.output.get("answer", pred.output)),
            schema,
        )

    if name in ("wer", "cer"):
        return error_rate(
            _output_text(pred) or "", _reference_text(sample), characters=name == "cer"
        )

    if name in (
        "precision_at_k",
        "recall_at_k",
        "hit_at_k",
        "success_at_k",
        "mrr",
        "map",
        "ndcg_at_k",
        "r_precision",
        "evidence_coverage",
    ):
        assert pred.contexts is not None
        ordered = [
            item.chunk_id for item in sorted(pred.contexts, key=lambda item: item.rank)
        ]

        return ranking_metric(
            name, ordered, set(ref["relevant_chunk_ids"]), int(settings.get("k", 10))
        )

    if name == "citation_validity":
        available = {item.chunk_id for item in pred.contexts or []}

        return citation_validity(pred.output.get("citations", []), available)

    if name == "agent_state_success":
        assert pred.trace is not None

        if pred.trace.observed_state is None:
            raise ValueError(
                "observed sandbox state absent; an agent claim is not verification"
            )

        return state_success(ref["expected_state"], pred.trace.observed_state)

    if name == "unauthorized_tool_rate":
        assert pred.trace is not None

        return unauthorized_tool_rate(pred.trace, set(ref["allowed_tools"]))

    if name == "trajectory_check":
        assert pred.trace is not None

        return trajectory_check(pred.trace, ref["expected_tools"])

    if name == "attack_success":
        oracle = ref["success_oracle"]

        if not isinstance(oracle, dict) or "expected_value" not in oracle:
            raise ValueError(
                "attack success requires an explicit expected_value oracle"
            )

        observed = pred.output["oracle_result"]

        if not isinstance(observed, dict) or observed.get("verified") is not True:
            raise ValueError(
                "attack observation must be verified; a detector flag is insufficient"
            )

        return float(observed.get("value") == oracle["expected_value"])

    raise ValueError(f"no sample scorer for {name}")


class EvaluationEngine:
    """Compute per-sample evidence, corpus metrics and honest coverage."""

    def evaluate(
        self,
        samples: list[EvalSample],
        predictions: list[PredictionRecord],
        metrics: list[MetricSpec],
        *,
        name: str = "evaluation",
        source_mode: str = "offline",
        is_fixture: bool = False,
        strict: bool = False,
        config: dict[str, Any] | None = None,
    ) -> RunResult:
        preflight(samples, metrics)
        aligned, extras = SampleAligner().align(samples, predictions)

        if strict and (
            extras or any(pred is None or pred.error for _, pred in aligned)
        ):
            raise InputError("strict alignment: extra, missing or failed predictions")

        errors = [f"extra prediction not in dataset: {sid}" for sid in extras]
        errors += [
            f"missing prediction: {s.sample_id}" for s, p in aligned if p is None
        ]
        errors += [
            f"target error {s.sample_id}: {p.error}"
            for s, p in aligned
            if p is not None and p.error
        ]
        results: list[MetricResult] = []
        scores: list[SampleScore] = []
        registry = MetricRegistry()

        for spec in metrics:
            descriptor = registry.get(spec.name)
            eligible: list[tuple[EvalSample, PredictionRecord]] = []
            missing = 0
            failed = 0

            for sample, pred in aligned:
                if not has_reference(sample, descriptor.required_reference):
                    scores.append(
                        SampleScore(
                            sample_id=sample.sample_id,
                            metric=spec.name,
                            status="insufficient_reference",
                            reason=f"requires {descriptor.required_reference}",
                        )
                    )
                    continue

                if pred is None:
                    missing += 1
                    scores.append(
                        SampleScore(
                            sample_id=sample.sample_id,
                            metric=spec.name,
                            status="skipped",
                            reason="missing prediction",
                        )
                    )
                elif not _has_prediction(
                    pred, descriptor.family, descriptor.required_prediction
                ):
                    failed += 1
                    code = "target_error" if pred.error else "insufficient_input"
                    scores.append(
                        SampleScore(
                            sample_id=sample.sample_id,
                            metric=spec.name,
                            status=code,
                            reason=pred.error
                            or f"requires {descriptor.required_prediction}",
                        )
                    )
                else:
                    eligible.append((sample, pred))

            n_scored = 0
            value: float | None = None
            details: dict[str, Any] = {
                "metric_version": descriptor.version,
                "orientation": descriptor.orientation,
            }
            good: list[tuple[EvalSample, PredictionRecord]] = []
            values: list[float] = []

            for sample, pred in eligible:
                if descriptor.family in ("classification", "performance"):
                    good.append((sample, pred))
                    continue

                try:
                    score = _sample_metric(spec.name, sample, pred, spec.settings)

                    if not np.isfinite(score):
                        raise ValueError("score is not finite")
                except (ValueError, TypeError, KeyError, AssertionError) as exc:
                    failed += 1
                    scores.append(
                        SampleScore(
                            sample_id=sample.sample_id,
                            metric=spec.name,
                            status="insufficient_input",
                            reason=str(exc),
                        )
                    )
                else:
                    good.append((sample, pred))
                    values.append(float(score))
                    scores.append(
                        SampleScore(
                            sample_id=sample.sample_id,
                            metric=spec.name,
                            value=float(score),
                        )
                    )

            if descriptor.family == "classification" and good:
                try:
                    value, extra = calculate_classification(
                        spec.name, good, spec.settings
                    )
                    details.update(extra)

                    if value is not None and not np.isfinite(value):
                        raise ValueError("invalid aggregate score")
                except (ValueError, TypeError, KeyError) as exc:
                    failed += len(good)

                    for sample, _ in good:
                        scores.append(
                            SampleScore(
                                sample_id=sample.sample_id,
                                metric=spec.name,
                                status="insufficient_input",
                                reason=str(exc),
                            )
                        )

                    good = []
                else:
                    if descriptor.family == "classification":
                        for sample, pred in good:
                            assert sample.reference is not None

                            scores.append(
                                SampleScore(
                                    sample_id=sample.sample_id,
                                    metric=spec.name,
                                    value=float(
                                        str(sample.reference["label"])
                                        == str(pred.output["label"])
                                    )
                                    if spec.name == "accuracy"
                                    else None,
                                    evidence={
                                        "predicted_label": str(pred.output["label"]),
                                        "reference_label": str(
                                            sample.reference["label"]
                                        ),
                                        "correct": str(sample.reference["label"])
                                        == str(pred.output["label"]),
                                    },
                                )
                            )
            elif descriptor.family == "performance" and good:
                durations = [
                    pred.timing.latency_ms
                    for _, pred in good
                    if pred.timing is not None
                ]

                if spec.name == "latency_mean":
                    value = float(np.mean(durations))
                else:
                    quantile = int(spec.name.removeprefix("latency_p"))
                    value = latency_percentile(durations, quantile)

                details["unit"] = "ms"
                details["timing_provenance"] = sorted(
                    {
                        pred.timing.provenance
                        for _, pred in good
                        if pred.timing is not None
                    }
                )
            elif values:
                value = float(np.mean(values))

            n_scored = len(good)
            n_eligible = sum(
                has_reference(s, descriptor.required_reference) for s, _ in aligned
            )
            missing = sum(pred is None for _, pred in aligned)
            coverage = n_scored / len(aligned)
            status = (
                "completed"
                if n_scored == len(aligned)
                else "partial"
                if n_scored
                else "insufficient_reference"
                if n_eligible == 0
                else "insufficient_input"
            )

            if spec.name == "confusion_matrix" and n_scored:
                status = "completed" if n_scored == len(aligned) else "partial"

            results.append(
                MetricResult(
                    name=spec.name,
                    value=value,
                    count=n_scored,
                    coverage=coverage,
                    status=status,
                    n_total=len(aligned),
                    n_eligible=n_eligible,
                    n_scored=n_scored,
                    n_failed=failed,
                    n_missing=missing,
                    details=details,
                )
            )

        from omni_eval import __version__

        manifest = RunManifest(
            run_id=uuid4().hex,
            name=name,
            status="completed"
            if not errors and all(m.status == "completed" for m in results)
            else "partial",
            source_mode=source_mode,
            dataset_hash=hash_canonical(
                [
                    s.model_dump(mode="json")
                    for s in sorted(samples, key=lambda s: s.sample_id)
                ]
            ),
            predictions_hash=hash_canonical(
                [
                    p.model_dump(mode="json")
                    for p in sorted(predictions, key=lambda p: p.sample_id)
                ]
            ),
            config_hash=hash_canonical(
                config or {"metrics": [m.model_dump(mode="json") for m in metrics]}
            ),
            package_version=__version__,
            is_fixture=is_fixture,
            metadata={
                "metrics": [metric.model_dump(mode="json") for metric in metrics],
                **({"marker": "DRY RUN / SYNTHETIC"} if is_fixture else {}),
            },
        )

        return RunResult(
            manifest=manifest,
            metrics=results,
            sample_scores=scores,
            samples=samples,
            predictions=predictions,
            errors=errors,
        )
