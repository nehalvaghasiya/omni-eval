"""Optional Parquet reader; never eagerly imports Arrow."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any


class Parquet:
    def read(self, path: str | Path) -> Iterator[dict[str, Any]]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Parquet support requires the 'parquet' extra. See "
                "https://github.com/nehalvaghasiya/omni-eval#installation"
            ) from exc
        reader = pq.ParquetFile(str(path))
        for batch in reader.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                if not isinstance(row, dict):
                    raise ValueError("invalid Parquet row")
                yield row
