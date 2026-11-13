"""Size-bounded UTF-8 JSONL with line-numbered errors."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from omni_eval.exceptions import InputError

T = TypeVar("T", bound=BaseModel)


def load_jsonl(
    path: str | Path, model: type[T], *, max_line_bytes: int = 4_194_304
) -> list[T]:
    records: list[T] = []
    seen: set[str] = set()

    try:
        with Path(path).open("rb") as source:
            for lineno, line in enumerate(
                iter(lambda: source.readline(max_line_bytes + 1), b""), 1
            ):
                if len(line) > max_line_bytes or (
                    line
                    and not line.endswith(b"\n")
                    and len(line) == max_line_bytes + 1
                ):
                    raise InputError(f"{path}:{lineno}: line exceeds size limit")

                if not line.strip():
                    raise InputError(f"{path}:{lineno}: blank JSONL line")

                try:
                    item = model.model_validate(json.loads(line.decode("utf-8")))
                except (ValueError, UnicodeError, ValidationError) as exc:
                    raise InputError(f"{path}:{lineno}: {exc}") from exc

                sample_id = getattr(item, "sample_id", None)

                if sample_id is not None:
                    if sample_id in seen:
                        raise InputError(
                            f"{path}:{lineno}: duplicate sample_id {sample_id!r}"
                        )

                    seen.add(sample_id)

                records.append(item)
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc

    if not records:
        raise InputError(f"empty JSONL: {path}")

    return records


def write_jsonl(path: str | Path, rows: Iterator[BaseModel] | list[BaseModel]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as sink:
        for item in rows:
            sink.write(item.model_dump_json(exclude_none=True) + "\n")
