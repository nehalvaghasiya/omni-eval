"""Small safe field-path subset; no Python evaluation or JSONPath filters."""

import re
from typing import Any

from omni_eval.exceptions import InputError

_SEGMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")


def get_path(data: Any, path: str) -> Any:
    if not path.startswith("$."):
        raise InputError(f"field selector must start with $.: {path!r}")

    segments = path[2:].split(".")

    if not segments or any(not _SEGMENT.fullmatch(part) for part in segments):
        raise InputError(f"unsafe field selector: {path!r}")

    value = data

    for part in segments:
        try:
            value = value[int(part)] if isinstance(value, list) else value[part]
        except (IndexError, KeyError, TypeError) as exc:
            raise InputError(f"field selector {path!r} missing at {part!r}") from exc

    return value


def map_template(template: Any, sample: dict[str, Any]) -> Any:
    if isinstance(template, str) and template.startswith("$."):
        return get_path(sample, template)

    if isinstance(template, dict):
        return {key: map_template(value, sample) for key, value in template.items()}

    if isinstance(template, list):
        return [map_template(value, sample) for value in template]

    return template
