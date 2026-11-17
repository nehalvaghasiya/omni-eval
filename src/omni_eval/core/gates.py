"""Three-valued CI threshold policy, no missing-metric pass."""

from omni_eval.types.gate import GateRule
from omni_eval.types.gate_result import GateResult
from omni_eval.types.run import Run


class GateEngine:
    def check(self, run: Run, rules: list[GateRule]) -> list[GateResult]:
        out = []
        for rule in rules:
            if rule.min_pairs:
                out.append(
                    GateResult(
                        rule=rule.metric,
                        status="inconclusive",
                        reason="absolute run gates have no paired comparison evidence",
                    )
                )
                continue
            try:
                score = run.metric(rule.metric)
            except KeyError:
                out.append(
                    GateResult(rule=rule.metric, status="inconclusive", reason="metric missing")
                )
                continue
            if (
                score.value is None
                or score.status not in {"complete", "partial"}
                or score.coverage < rule.min_coverage
            ):
                out.append(
                    GateResult(
                        rule=rule.metric,
                        status="inconclusive",
                        reason="unavailable metric or insufficient coverage",
                        value=score.value,
                    )
                )
                continue
            if rule.op == "gte":
                passed = score.value >= rule.value
            elif rule.op == "lte":
                passed = score.value <= rule.value
            else:
                raise ValueError(f"unsupported absolute gate operation {rule.op!r}")
            out.append(
                GateResult(
                    rule=rule.metric,
                    status="pass" if passed else "fail",
                    reason="threshold satisfied" if passed else "threshold breached",
                    value=score.value,
                )
            )
        return out
