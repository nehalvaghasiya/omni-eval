"""Reuse stored predictions only, never infer again."""

from omni_eval.sources.base import Source
from omni_eval.storage.store import Store
from omni_eval.types.case import Case
from omni_eval.types.config import Config


class ReplaySource(Source):
    def acquire(self, cases: list[Case], config: Config) -> list[Case]:
        if not config.data:
            raise ValueError("replay requires run path")
        saved = Store().load(config.data)
        if not saved.cases:
            raise ValueError("run did not retain cases; cannot replay")
        return saved.cases
