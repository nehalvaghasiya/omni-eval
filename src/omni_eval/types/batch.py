"""Immutable scoring view of canonical cases."""

from pydantic import BaseModel, ConfigDict, Field

from omni_eval.types.case import Case


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task: str
    cases: list[Case] = Field(default_factory=list)

    def complete(self) -> list[Case]:
        return [c for c in self.cases if c.actual is not None and c.error is None]

    def select(self, needs: frozenset[str]) -> list[Case]:
        return [c for c in self.complete() if all(c.get(n) is not None for n in needs)]
