from omni_eval.core.metric_descriptor import (
    MetricDescriptor,
    MetricFamily,
    MetricOrientation,
)
from omni_eval.exceptions import InputError

# Canonical static catalog. Only implemented metrics are registered.
_METRICS: dict[str, tuple[MetricFamily, str | None, str, MetricOrientation, str]] = {}


def _register(
    names: str,
    family: MetricFamily,
    reference: str | None,
    prediction: str,
    orientation: MetricOrientation = "higher_is_better",
    aggregation: str = "mean",
) -> None:
    for name in names.split():
        _METRICS[name] = (family, reference, prediction, orientation, aggregation)


_register(
    "accuracy macro_f1 micro_f1 weighted_f1 macro_precision macro_recall weighted_precision weighted_recall balanced_accuracy mcc cohen_kappa specificity confusion_matrix",
    "classification",
    "label",
    "label",
    aggregation="corpus",
)
_register(
    "brier_score ece mce top_k_accuracy log_loss auroc average_precision",
    "classification",
    "label",
    "probabilities",
    "lower_is_better",
    "corpus",
)
_register(
    "top_k_accuracy auroc average_precision",
    "classification",
    "label",
    "probabilities",
    "higher_is_better",
    "corpus",
)
_register(
    "exact_match normalized_exact_match token_precision token_recall token_f1",
    "text",
    "text",
    "text",
)
_register("json_validity", "text", None, "output")
_register("json_schema_validity", "text", "schema", "output")
_register("wer cer", "text", "text", "text", "lower_is_better")
_register(
    "precision_at_k recall_at_k hit_at_k success_at_k mrr map ndcg_at_k r_precision",
    "retrieval",
    "relevant_chunk_ids",
    "contexts",
)
_register("citation_validity", "rag", None, "contexts")
_register("evidence_coverage", "rag", "relevant_chunk_ids", "contexts")
_register("agent_state_success", "agent", "expected_state", "trace")
_register(
    "unauthorized_tool_rate", "agent", "allowed_tools", "trace", "lower_is_better"
)
_register("trajectory_check", "agent", "expected_tools", "trace")
_register(
    "latency_mean latency_p50 latency_p95 latency_p99",
    "performance",
    None,
    "timing",
    "lower_is_better",
    "corpus",
)
_register(
    "attack_success", "security", "success_oracle", "oracle_result", "lower_is_better"
)


class MetricRegistry:
    """Explicit native metric registry. Optional frameworks are never imported."""

    def list(self) -> list[MetricDescriptor]:
        return [self.get(name) for name in sorted(_METRICS)]

    def get(self, name: str) -> MetricDescriptor:
        if name not in _METRICS:
            raise InputError(
                f"unknown or unavailable metric {name!r}; see 'omni-eval metrics list'"
            )

        family, reference, prediction, orientation, aggregation = _METRICS[name]
        unbounded = {
            "wer",
            "cer",
            "log_loss",
            "latency_mean",
            "latency_p50",
            "latency_p95",
            "latency_p99",
        }
        bounds = (
            None
            if name in unbounded or name == "confusion_matrix"
            else (-1.0, 1.0)
            if name in {"mcc", "cohen_kappa"}
            else (0.0, 2.0)
            if name == "brier_score"
            else (0.0, 1.0)
        )

        if name == "confusion_matrix":
            orientation = "informational"

        return MetricDescriptor(
            name,
            family,
            reference,
            prediction,
            orientation,
            score_range=bounds,
            aggregation=aggregation,
        )
