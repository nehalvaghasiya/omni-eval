"""Privacy-aware group views; suppress small support slices."""

from omni_eval.types.batch import Batch
from omni_eval.types.case import Case


class Slicer:
    def group(self, batch: Batch, key: str, *, min_count: int = 5) -> dict[str, Batch]:
        if min_count < 2:
            raise ValueError("slice minimum support must be at least two")
        groups: dict[str, list[Case]] = {}
        for case in batch.cases:
            value = case.meta.get(key)
            if value is None:
                continue
            groups.setdefault(str(value), []).append(case)
        return {
            name: Batch(task=batch.task, cases=cases)
            for name, cases in groups.items()
            if len(cases) >= min_count
        }
