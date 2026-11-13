import json
from typing import Any


def json_validity(output: Any) -> float:
    if isinstance(output, (dict, list)):
        return 1.0

    if not isinstance(output, str):
        return 0.0

    try:
        json.loads(output)
    except (ValueError, TypeError):
        return 0.0

    return 1.0
