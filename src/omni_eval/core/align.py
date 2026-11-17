"""Stable ID join; never positional alignment."""

from collections.abc import Iterable
from typing import Any

from omni_eval.io.mapping import Mapper
from omni_eval.types.case import Case


class Aligner:
    def join(
        self,
        samples: Iterable[Case],
        predictions: Iterable[dict[str, Any]],
        *,
        override: bool = False,
    ) -> list[Case]:
        supplied: dict[str, dict[str, Any]] = {}
        mapper = Mapper()
        for row in predictions:
            sid, observed = mapper.to_actual(row)
            if sid in supplied:
                raise ValueError(f"duplicate prediction id {sid!r}")
            supplied[sid] = observed
        result = []
        ids = set()
        for sample in samples:
            if sample.id in ids:
                raise ValueError(f"duplicate case id {sample.id!r}")
            ids.add(sample.id)
            if sample.id in supplied:
                if sample.actual is not None and (not override):
                    raise ValueError(f"case {sample.id!r} contains actual and separate prediction")
                result.append(sample.model_copy(update=supplied[sample.id]))
            else:
                result.append(sample)
        extra = set(supplied) - ids
        if extra:
            raise ValueError(f"predictions contain unknown case IDs: {sorted(extra)[:5]}")
        return [Case.model_validate(s.model_dump()) for s in result]
