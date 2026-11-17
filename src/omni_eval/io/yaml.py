"""Safe YAML configuration parser."""

from pathlib import Path
from typing import Any

import yaml


def load(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        obj = yaml.safe_load(file)
    if not isinstance(obj, dict):
        raise ValueError("YAML root must be a mapping")
    return obj
