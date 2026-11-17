"""Bounded UTF-8 JSONL reader and writer."""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


class Jsonl:
    def read(self, path: str | Path, *, max_line: int = 2000000) -> Iterator[dict[str, Any]]:
        with Path(path).open("r", encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    raise ValueError(f"{path}:{number}: blank JSONL line")
                if len(line.encode("utf-8")) > max_line:
                    raise ValueError(f"{path}:{number}: JSONL line exceeds size limit")
                try:
                    obj = json.loads(
                        line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))
                    )
                except (ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
                if not isinstance(obj, dict):
                    raise ValueError(f"{path}:{number}: expected a JSON object")
                yield obj

    def write(self, path: str | Path, items: Iterable[dict[str, Any]]) -> None:
        with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
            for item in items:
                stream.write(
                    json.dumps(item, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
                )
