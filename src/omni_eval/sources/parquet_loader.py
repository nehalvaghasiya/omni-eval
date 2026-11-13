"""Optional Parquet reader with strict schema validation."""

from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from omni_eval.exceptions import DependencyUnavailable, InputError

T = TypeVar("T", bound=BaseModel)


def load_parquet(path: str | Path, model: type[T]) -> list[T]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise DependencyUnavailable(
            "install omni-eval[parquet] for Parquet input"
        ) from exc

    rows = parquet.read_table(path).to_pylist()

    if not rows:
        raise InputError("empty Parquet input")

    result = [model.model_validate(row) for row in rows]
    ids = [getattr(r, "sample_id", None) for r in result]

    if len(ids) != len(set(ids)):
        raise InputError("duplicate sample_id in Parquet input")

    return result
