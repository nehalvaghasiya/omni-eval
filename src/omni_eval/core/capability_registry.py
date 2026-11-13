"""Preflight validates required references before endpoint acquisition."""

from omni_eval.core.metric_registry import MetricRegistry
from omni_eval.exceptions import InputError
from omni_eval.schemas.eval_sample import EvalSample
from omni_eval.schemas.metric_spec import MetricSpec


def has_reference(sample: EvalSample, required: str | None) -> bool:
    if required is None:
        return True

    ref = sample.reference or {}

    if required == "text":
        return any(isinstance(ref.get(key), str) for key in ("answer", "text", "value"))

    if required == "schema":
        return isinstance(ref.get("schema", sample.input.get("schema")), dict)

    if required == "relevant_chunk_ids":
        return bool(ref.get("relevant_chunk_ids"))

    return required in ref and ref[required] is not None


def preflight(samples: list[EvalSample], metrics: list[MetricSpec]) -> None:
    if not samples:
        raise InputError("evaluation requires at least one sample")

    registry = MetricRegistry()

    for spec in metrics:
        descriptor = registry.get(spec.name)

        if not any(
            has_reference(sample, descriptor.required_reference) for sample in samples
        ):
            raise InputError(
                f"{spec.name} requires reference.{descriptor.required_reference}; none of the samples satisfy it"
            )
