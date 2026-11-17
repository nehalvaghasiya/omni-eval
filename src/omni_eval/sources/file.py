"""Read inline/split saved predictions without calling a model."""

from collections.abc import Iterator
from typing import Any

from omni_eval.core.align import Aligner
from omni_eval.io.jsonl import Jsonl
from omni_eval.io.mapping import Mapper
from omni_eval.sources.base import Source
from omni_eval.types.case import Case
from omni_eval.types.config import Config


class FileSource(Source):
    def acquire(self, cases: list[Case], config: Config) -> list[Case]:
        if not config.predictions:
            return cases
        rows = list(self._rows(config.predictions, config))
        return Aligner().join(cases, rows, override=config.actual_source == "predictions")

    def _rows(self, path: str, config: Config) -> Iterator[dict[str, Any]]:
        if path.endswith(".csv"):
            from omni_eval.io.csv import Csv

            return Csv().read(path, config.columns, json_columns=set(config.json_columns))
        if path.endswith(".parquet"):
            from omni_eval.io.parquet import Parquet

            return Parquet().read(path)
        return Jsonl().read(path)

    def read(self, config: Config) -> list[Case]:
        if not config.data:
            raise ValueError("data file is required")
        mapper = Mapper()
        samples = list(map(mapper.to_case, self._rows(config.data, config)))
        if not samples:
            raise ValueError("dataset contains zero cases")
        ids = [c.id for c in samples]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate case id in dataset")
        return self.acquire(samples, config)
