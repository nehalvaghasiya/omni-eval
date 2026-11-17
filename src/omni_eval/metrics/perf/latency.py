"""Observed latency quantiles, not evaluation parsing time."""

from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.types.batch import Batch
from omni_eval.types.score import Score


class Latency(Metric):
    name = "latency_p95"
    needs = frozenset({"timing"})
    direction = "down"
    unit = "milliseconds"
    tasks = frozenset(
        {
            "classification",
            "text",
            "retrieval",
            "rag",
            "agent",
            "mcp",
            "security",
            "audio",
            "vision",
            "code",
            "sql",
        }
    )
    level = "corpus"

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        selected, points, failed = self.eligible(batch)
        scopes = {(c.timing.scope, c.timing.source) for c in selected if c.timing}
        if len(scopes) > 1:
            raise ValueError("latencies use incompatible scopes/provenance")
        values = sorted(c.timing.latency_ms for c in selected if c.timing is not None)
        if not values:
            return self._result(batch, [], points, failed)
        if self.name == "latency_mean":
            value = sum(values) / len(values)
        else:
            q = float(self.name.replace("latency_p", "")) / 100
            index = q * (len(values) - 1)
            lo = int(index)
            hi = min(len(values) - 1, lo + 1)
            value = values[lo] + (values[hi] - values[lo]) * (index - lo)
        return self._result(
            batch,
            [0.0] * len(values),
            points,
            failed,
            value=value,
            details={
                "scope": list(scopes)[0][0],
                "source": list(scopes)[0][1],
                "quantile": "linear",
            },
        )
