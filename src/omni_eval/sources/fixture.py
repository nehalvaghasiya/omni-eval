"""Deterministic, explicitly synthetic prediction acquisition."""

from omni_eval.sources.base import Source
from omni_eval.types.case import Case
from omni_eval.types.config import Config


class FixtureSource(Source):
    def acquire(self, cases: list[Case], config: Config) -> list[Case]:
        return cases
