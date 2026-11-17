"""Only parse Python syntax; never execute generated code."""

import ast
from typing import Any

from omni_eval.metrics.base import Metric
from omni_eval.metrics.data import answer
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class Syntax(Metric):
    name = "code_valid"
    tasks = frozenset({"code"})
    needs = frozenset({"actual"})

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        code = answer(case)
        if code is None:
            raise ValueError("code_valid needs Python source text")
        if settings.get("language", "python") != "python":
            raise ValueError("only Python AST syntax supported")
        try:
            ast.parse(code)
            return 1.0
        except SyntaxError:
            return 0.0

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
