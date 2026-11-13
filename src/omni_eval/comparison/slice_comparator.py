from collections import defaultdict

from omni_eval.schemas.run_result import RunResult


class SliceComparator:
    """Report per-slice mean only where privacy minimum support is met."""

    def summarize(
        self, run: RunResult, metric: str, key: str = "slice", min_support: int = 5
    ) -> dict[str, dict[str, float | int]]:
        groups: dict[str, list[float]] = defaultdict(list)
        samples = {s.sample_id: s for s in run.samples}

        for score in run.sample_scores:
            if (
                score.metric != metric
                or score.value is None
                or score.sample_id not in samples
            ):
                continue

            label = samples[score.sample_id].metadata.get(key)

            if label is not None:
                groups[str(label)].append(score.value)

        return {
            group: {"mean": sum(values) / len(values), "n": len(values)}
            for group, values in groups.items()
            if len(values) >= min_support
        }
