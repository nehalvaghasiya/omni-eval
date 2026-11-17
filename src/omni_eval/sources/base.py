"""One stable source interface for all prediction acquisition."""

from abc import ABC, abstractmethod

from omni_eval.types.case import Case
from omni_eval.types.config import Config


class Source(ABC):
    @abstractmethod
    def acquire(self, cases: list[Case], config: Config) -> list[Case]:
        """Return the same case IDs with acquired predictions."""
