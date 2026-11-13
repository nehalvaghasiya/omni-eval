import json
from typing import Any

import jsonschema


def schema_compliance(output: Any, schema: dict[str, Any]) -> float:
    """Validate against a declared JSON Schema (not a judge interpretation)."""
    jsonschema.Draft202012Validator.check_schema(schema)

    if isinstance(output, str):
        try:
            output = json.loads(output)
        except ValueError:
            return 0.0

    return float(jsonschema.Draft202012Validator(schema).is_valid(output))
