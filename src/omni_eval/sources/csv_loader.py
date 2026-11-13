"""Explicit CSV column mapping; nested JSON is opt-in."""

import csv
import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from omni_eval.exceptions import InputError

T = TypeVar("T", bound=BaseModel)


def load_csv(path: str | Path, model: type[T], mapping: dict[str, str]) -> list[T]:
    if not mapping:
        raise InputError(
            "CSV ingestion requires explicit target_field: source_column mapping"
        )

    records: list[T] = []
    seen: set[str] = set()

    try:
        with Path(path).open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)

            if reader.fieldnames is None or not set(mapping.values()).issubset(
                reader.fieldnames
            ):
                raise InputError("CSV mapping references columns missing from header")

            for index, row in enumerate(reader, 2):
                payload: dict[str, Any] = {}

                for target, column in mapping.items():
                    value = row[column]

                    if value is None:
                        raise InputError(f"CSV row {index}: missing column {column}")

                    if (
                        target
                        in {
                            "input",
                            "output",
                            "reference",
                            "metadata",
                            "probabilities",
                            "contexts",
                            "trace",
                            "timing",
                        }
                        and value
                    ):
                        try:
                            payload[target] = json.loads(value)
                        except json.JSONDecodeError as exc:
                            raise InputError(
                                f"CSV row {index}: {column}: invalid JSON"
                            ) from exc
                    else:
                        payload[target] = value

                try:
                    item = model.model_validate(payload)
                except ValidationError as exc:
                    raise InputError(f"CSV row {index}: {exc}") from exc

                sid = getattr(item, "sample_id", None)

                if sid is not None and sid in seen:
                    raise InputError(f"CSV row {index}: duplicate sample_id {sid!r}")

                if sid is not None:
                    seen.add(sid)

                records.append(item)
    except OSError as exc:
        raise InputError(f"cannot read CSV {path}: {exc}") from exc

    if not records:
        raise InputError(f"empty CSV: {path}")

    return records
