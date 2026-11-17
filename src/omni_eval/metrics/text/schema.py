"""JSON Schema conformance without executing generated content."""

import json
from typing import Any

import jsonschema
from referencing import Registry
from referencing.exceptions import Unresolvable

from omni_eval.metrics.base import Metric
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.score import Score


class SchemaValid(Metric):
    version = "2"
    name = "schema_valid"
    tasks = frozenset({"text", "rag", "agent", "code"})
    needs = frozenset({"actual"})

    def _point(self, case: Case, settings: dict[str, Any]) -> float:
        schema = settings.get("schema")
        if schema is None and isinstance(case.expected, dict):
            schema = case.expected.get("schema")
        if not isinstance(schema, dict):
            raise ValueError("schema_valid requires expected.schema or config schema")
        try:
            jsonschema.Draft202012Validator.check_schema(schema)
        except jsonschema.SchemaError as exc:
            raise ValueError("invalid JSON Schema configuration") from exc
        value = case.actual
        if isinstance(value, str):
            try:
                value = json.loads(value)
                json.dumps(value, allow_nan=False)
            except ValueError:
                return 0.0
        else:
            try:
                json.dumps(value, allow_nan=False)
            except (TypeError, ValueError):
                return 0.0
        try:
            # An empty registry disables remote schema retrieval. Local #/$defs
            # references still work; an unresolved reference is a scoring error.
            return float(
                jsonschema.Draft202012Validator(schema, registry=Registry()).is_valid(value)
            )
        except Unresolvable as exc:
            raise ValueError(
                "JSON Schema reference is unresolved; only local references are supported"
            ) from exc

    def compute(self, batch: Batch, settings: dict[str, Any] | None = None) -> Score:
        return self.sample_score(batch, settings)
