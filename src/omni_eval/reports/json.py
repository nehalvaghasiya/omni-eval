"""Machine-readable full-precision report; never recompute scores."""

from typing import Any

from omni_eval.types.run import Run


class Json:
    def render(self, run: Run) -> dict[str, Any]:
        return {
            "id": run.id,
            "task": run.task,
            "fixture": run.fixture,
            "manifest": run.manifest,
            "scores": [s.model_dump(mode="json") for s in run.scores],
        }
