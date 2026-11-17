"""Only compare compatible runs, with paired differences where meaningful."""

import random

from omni_eval.types.compare import CompareResult
from omni_eval.types.run import Run


class Comparator:
    def compare(
        self,
        left: Run,
        right: Run,
        metrics: list[str] | None = None,
        *,
        seed: int = 42,
        confidence: float = 0.95,
        iterations: int = 1000,
    ) -> CompareResult:
        if left.task != right.task:
            raise ValueError("cannot compare different tasks")
        if not 0 < confidence < 1 or iterations < 100:
            raise ValueError("invalid confidence/iterations")
        left_cases = {c.id: c for c in left.cases}
        right_cases = {c.id: c for c in right.cases}
        shared = sorted(left_cases.keys() & right_cases.keys())
        if not shared:
            raise ValueError("no common case IDs")
        for sid in shared:
            old_case = left_cases[sid]
            new_case = right_cases[sid]
            if old_case.expected != new_case.expected or old_case.input != new_case.input:
                raise ValueError(f"different reference/input for paired case {sid}")
        selected = metrics or sorted(
            set(s.name for s in left.scores) & set(s.name for s in right.scores)
        )
        result = CompareResult(baseline=left.id, candidate=right.id)
        rng = random.Random(seed)
        for name in selected:
            left_settings = left.config.get("limits", {}).get("metric_settings", {}).get(name, {})
            right_settings = right.config.get("limits", {}).get("metric_settings", {}).get(name, {})
            if left_settings != right_settings:
                raise ValueError(f"incompatible metric settings: {name}")
            a = left.metric(name)
            b = right.metric(name)
            if (a.version, a.source, a.unit, a.details.get("scope"), a.details.get("source")) != (
                b.version,
                b.source,
                b.unit,
                b.details.get("scope"),
                b.details.get("source"),
            ):
                raise ValueError(f"incompatible metric methodology: {name}")
            old = {p.id: p.value for p in a.points if p.value is not None}
            new = {p.id: p.value for p in b.points if p.value is not None}
            paired = [new[s] - old[s] for s in shared if s in old and s in new]
            result.paired_n[name] = len(paired)
            same_cases = set(left_cases) == set(right_cases)
            if not same_cases:
                if paired:
                    result.deltas[name] = sum(paired) / len(paired)
                    result.warnings.append(
                        f"{name}: paired subset delta, not a whole-dataset delta"
                    )
                else:
                    result.deltas[name] = None
                    result.warnings.append(f"{name}: corpus delta withheld for mismatched datasets")
            elif b.value is not None and a.value is not None:
                result.deltas[name] = b.value - a.value
                if len(paired) > 0 and len(paired) < len(shared):
                    result.warnings.append(
                        f"{name}: unequal scoring coverage may bias aggregate delta"
                    )
            else:
                result.deltas[name] = None
            if len(paired) >= 2:
                samples = [
                    sum(paired[rng.randrange(len(paired))] for _ in paired) / len(paired)
                    for _ in range(iterations)
                ]
                samples.sort()
                alpha = (1 - confidence) / 2
                result.intervals[name] = [
                    samples[int(alpha * (iterations - 1))],
                    samples[int((1 - alpha) * (iterations - 1))],
                ]
            elif len(paired) == 0:
                result.warnings.append(f"{name}: corpus-only metric; no paired bootstrap interval")
        if set(left_cases) != set(right_cases):
            result.warnings.append("comparison uses only shared sample IDs")
        return result
