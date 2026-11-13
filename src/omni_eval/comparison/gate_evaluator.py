from omni_eval.schemas.gate_result import GateResult
from omni_eval.schemas.metric_result import MetricResult


class GateEvaluator:
    """Three-outcome absolute gate: missing/partial evidence never passes."""

    def evaluate(
        self,
        metric: MetricResult | None,
        *,
        min_value: float | None = None,
        max_value: float | None = None,
        min_coverage: float = 1.0,
        gate_name: str = "quality",
    ) -> GateResult:
        if metric is None or metric.value is None:
            return GateResult(
                gate_name=gate_name, status="inconclusive", reason="metric unavailable"
            )

        if metric.coverage < min_coverage:
            return GateResult(
                gate_name=gate_name,
                status="inconclusive",
                reason=f"coverage {metric.coverage:.3f} below minimum {min_coverage:.3f}",
                value=metric.value,
            )

        if (min_value is not None and metric.value < min_value) or (
            max_value is not None and metric.value > max_value
        ):
            return GateResult(
                gate_name=gate_name,
                status="fail",
                reason="threshold violated",
                value=metric.value,
            )

        return GateResult(
            gate_name=gate_name,
            status="pass",
            reason="threshold and coverage satisfied",
            value=metric.value,
        )
