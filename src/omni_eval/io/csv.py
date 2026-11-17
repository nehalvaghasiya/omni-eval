"""CSV reader with explicit canonical field mappings."""

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class Csv:
    def read(
        self, path: str | Path, mapping: dict[str, str], *, json_columns: set[str] | None = None
    ) -> Iterator[dict[str, Any]]:
        if "id" not in mapping:
            raise ValueError("CSV mapping must define an 'id' column")
        with Path(path).open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                case: dict[str, Any] = {}
                for field, column in mapping.items():
                    if column not in row:
                        raise ValueError(f"missing CSV column {column!r}")
                    raw = row[column]
                    case[field] = (
                        json.loads(raw) if field in (json_columns or set()) and raw else raw
                    )
                yield case
